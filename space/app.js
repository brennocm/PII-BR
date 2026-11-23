// PII-BR no navegador: Pyodide roda o pacote pii_br (regras, fusão, mascaramento, tokenização)
// e o onnxruntime-web roda o modelo. O texto do usuário nunca é enviado pela rede: as únicas
// requisições baixam código (CDN), o pacote (pii_br.zip) e o modelo (Hugging Face).
"use strict";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const MODEL_CACHE = "pii-br-modelo";
const ENTITIES = [
  ["NOME", "Nome"], ["CPF", "CPF"], ["CNPJ", "CNPJ"], ["RG", "RG"], ["TELEFONE", "Telefone"],
  ["EMAIL", "E-mail"], ["ENDERECO", "Endereço"], ["CEP", "CEP"], ["DATA_NASC", "Data de nascimento"],
];
const NAMES = Object.fromEntries(ENTITIES);
const EXAMPLES = [
  "Oi, sou Maria Aparecida Souza, meu CPF é 123.456.789-09 e moro na Rua das Flores, 123, ap. 42, CEP 01310-100. Pode me ligar no (11) 98765-4321?",
  "manda o pix pro meu cpf 529.982.247-24 ou no zap 21 99876-5432 que eu confirmo",
  "CONTRATADA: Exemplo Serviços Ltda., inscrita no CNPJ sob o nº 12.ABC.345/01DE-35, representada por João Pedro Lima, RG 12.345.678-9, nascido em 3 de julho de 1985.",
  '2026-11-20T13:44:02Z INFO kyc aprovado nome="Fernanda Arantes" cpf=11144477735 email=fernanda.a@example.com cep=70040010',
  "Prezados, encaminho os dados do novo colaborador: Rafael Takeshi Nakamura, nascido em 14/06/1992, e-mail rafael.nakamura@example.com, celular +55 41 99652-0081.",
  "O pedido 2024000123 foi faturado em 15/11/2026 no valor de R$ 1.234,56. Dúvidas: central 0800 123 4567.",
];

const $ = (id) => document.getElementById(id);
const ui = {
  text: $("text"), run: $("run"), copy: $("copy"), status: $("status"), summary: $("summary"),
  masked: $("masked"), highlight: $("highlight"), legend: $("legend"), rows: $("span-rows"),
  progress: $("model-progress"), bar: $("model-bar"), barLabel: $("model-label"),
  threshold: $("threshold"), thresholdValue: $("threshold-value"),
};

let py = null; // funções da ponte em Python
let session = null; // sessão do modelo ONNX (null: só regras)
let modelState = "carregando"; // carregando | pronto | indisponível
let busy = false;

const fmtNumber = new Intl.NumberFormat("pt-BR");
const fmtDecimal = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const mb = (bytes) => fmtNumber.format(Math.round(bytes / 1e6));

// ------------------------------------------------------------------ interface

function buildControls() {
  const examples = $("examples");
  for (const example of EXAMPLES) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example;
    button.title = example;
    button.addEventListener("click", () => {
      ui.text.value = example;
      analyze();
    });
    li.append(button);
    examples.append(li);
  }
  const grid = $("entities");
  for (const [value, name] of ENTITIES) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "entity";
    input.value = value;
    input.checked = true;
    label.append(input, " " + name);
    grid.append(label);
  }
  ui.run.addEventListener("click", analyze);
  ui.text.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      analyze();
    }
  });
  ui.threshold.addEventListener("input", () => {
    ui.thresholdValue.textContent = fmtDecimal.format(Number(ui.threshold.value));
  });
  for (const input of document.querySelectorAll("input[name=strategy], input[name=entity], #threshold")) {
    input.addEventListener("change", () => { if (ui.text.value.trim()) analyze(); });
  }
  ui.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(ui.masked.textContent);
      ui.copy.textContent = "Copiado";
    } catch {
      ui.copy.textContent = "Não deu para copiar";
    }
    setTimeout(() => { ui.copy.textContent = "Copiar"; }, 1500);
  });
}

function setStatus(message, warn = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("warn", warn);
}

function modelStatus() {
  if (modelState === "pronto") return "Modelo e regras prontos. Tudo roda nesta página.";
  if (modelState === "indisponível") {
    return "Modelo indisponível agora: só regras (CPF, CNPJ, telefone, e-mail, CEP). Nomes, endereços, RG e datas precisam do modelo.";
  }
  return "Regras prontas. O modelo ainda está baixando: por enquanto só CPF, CNPJ, telefone, e-mail e CEP.";
}

function render(result, text) {
  const { spans } = result;
  ui.masked.textContent = result.masked;
  ui.copy.disabled = !result.masked;

  ui.highlight.replaceChildren();
  let last = 0;
  const counts = {};
  for (const span of spans) {
    if (span.start > last) ui.highlight.append(text.slice(last, span.start));
    const key = span.valid === false ? "ALERTA" : span.label;
    const mark = document.createElement("mark");
    mark.style.setProperty("--c", `var(--e-${key})`);
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = span.valid === false ? `${span.label} inválido` : span.label;
    mark.append(text.slice(span.start, span.end), tag);
    ui.highlight.append(mark);
    last = span.end;
    counts[span.label] = (counts[span.label] || 0) + 1;
  }
  if (last < text.length) ui.highlight.append(text.slice(last));

  ui.legend.replaceChildren(...Object.entries(counts).map(([label, n]) => {
    const li = document.createElement("li");
    li.style.setProperty("--c", `var(--e-${label})`);
    li.textContent = `${NAMES[label]} ${n}`;
    return li;
  }));

  ui.rows.replaceChildren(...spans.map((span) => {
    const tr = document.createElement("tr");
    const doc = span.valid === null ? "—" : span.valid ? "válido" : "dígito inválido";
    const cells = [NAMES[span.label], span.text, span.source, fmtDecimal.format(span.score), doc];
    cells.forEach((value, i) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (i === 1) td.className = "text";
      tr.append(td);
    });
    return tr;
  }));

  const alerts = spans.filter((s) => s.valid === false).length;
  ui.summary.replaceChildren();
  const total = spans.length;
  ui.summary.append(
    total === 0 ? "Nenhum dado pessoal encontrado."
      : total === 1 ? "1 dado pessoal encontrado." : `${fmtNumber.format(total)} dados pessoais encontrados.`,
  );
  if (alerts) {
    const alert = document.createElement("span");
    alert.className = "alert";
    alert.textContent = ` ${alerts} com dígito verificador inválido, mascarado${alerts > 1 ? "s" : ""} mesmo assim: pode ser erro de digitação num documento real.`;
    ui.summary.append(alert);
  }
  if (result.cut) {
    ui.summary.append(` Só os primeiros ${fmtNumber.format(result.chars)} caracteres foram analisados.`);
  }
}

// ------------------------------------------------------------------ análise

async function modelLogits(text) {
  const windows = JSON.parse(py.encode(text));
  const out = [];
  for (const ids of windows) {
    const n = ids.length;
    const tensor = (values) => new ort.Tensor("int64", BigInt64Array.from(values, BigInt), [1, n]);
    const result = await session.run({
      input_ids: tensor(ids),
      attention_mask: tensor(new Array(n).fill(1)),
      token_type_ids: tensor(new Array(n).fill(0)),
    });
    out.push(Array.from(result.logits.data));
  }
  return out;
}

async function analyze() {
  if (!py || busy) return;
  const text = ui.text.value;
  if (!text.trim()) {
    ui.summary.textContent = "Cole um texto ou escolha um exemplo.";
    return;
  }
  busy = true;
  ui.run.setAttribute("aria-busy", "true");
  ui.run.textContent = "Analisando…";
  await new Promise(requestAnimationFrame); // deixa o botão redesenhar antes do trabalho pesado
  try {
    const logits = session ? await modelLogits(text) : null;
    const request = {
      text,
      logits,
      strategy: document.querySelector("input[name=strategy]:checked").value,
      entities: [...document.querySelectorAll("input[name=entity]:checked")].map((i) => i.value),
      threshold: Number(ui.threshold.value),
    };
    render(JSON.parse(py.analyze(JSON.stringify(request))), text.slice(0, 20000));
  } catch (error) {
    console.error(error);
    ui.summary.textContent = "Não consegui analisar este texto. Recarregue a página e tente de novo.";
  } finally {
    busy = false;
    ui.run.removeAttribute("aria-busy");
    ui.run.textContent = "Mascarar texto";
  }
}

// ------------------------------------------------------------------ carga

async function fetchWithProgress(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // sem Cache Storage (aba anônima, por exemplo): baixa toda vez
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} em ${url}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    try {
      // Versões antigas do modelo saem do cache: só a URL atual (com o commit) fica.
      for (const old of await cache.keys()) if (old.url !== url) await cache.delete(old);
      await cache.put(url, new Response(bytes));
    } catch {
      // cota cheia: segue sem cache
    }
  }
  return bytes;
}

async function loadPython(config) {
  const pyodide = await loadPyodide();
  const archive = await fetch("pii_br.zip");
  if (!archive.ok) throw new Error("pacote pii_br.zip indisponível");
  pyodide.unpackArchive(await archive.arrayBuffer(), "zip");
  const bridge = pyodide.pyimport("bridge");
  const tokenizer = await fetch(config.tokenizer);
  if (tokenizer.ok) bridge.load_tokenizer(await tokenizer.text());
  return { encode: bridge.encode, analyze: bridge.analyze, hasTokenizer: tokenizer.ok };
}

async function loadModel(config) {
  ort.env.wasm.wasmPaths = ORT_DIST;
  ui.progress.hidden = false;
  const bytes = await fetchWithProgress(config.onnx, (received, total) => {
    ui.bar.value = total ? received / total : 0;
    ui.barLabel.textContent = total ? `${mb(received)} de ${mb(total)} MB` : `${mb(received)} MB`;
  });
  ui.barLabel.textContent = "Preparando o modelo…";
  const created = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  ui.progress.hidden = true;
  return created;
}

async function main() {
  buildControls();
  const config = await (await fetch("config.json")).json();
  // `base` só existe em teste local, para servir o modelo de uma pasta.
  const base = config.base || `https://huggingface.co/${config.model}/resolve/${config.revision}/`;
  config.onnx = base + "onnx/model_int8.onnx";
  config.tokenizer = base + "tokenizer.json";
  try {
    py = await loadPython(config);
  } catch (error) {
    console.error(error);
    setStatus("Não consegui carregar o Python no navegador. Recarregue a página.", true);
    return;
  }
  ui.run.disabled = false;
  if (!py.hasTokenizer) modelState = "indisponível";
  setStatus(modelStatus(), modelState === "indisponível");
  if (modelState === "indisponível") return;
  try {
    session = await loadModel(config);
    modelState = "pronto";
  } catch (error) {
    console.error(error);
    ui.progress.hidden = true;
    modelState = "indisponível";
  }
  setStatus(modelStatus(), modelState === "indisponível");
  if (session && ui.text.value.trim()) analyze(); // refaz com o modelo
}

main();
