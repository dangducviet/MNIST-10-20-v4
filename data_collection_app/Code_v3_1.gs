// Dán thay Code.gs. Dán HTML v3.1 vào file HTML tên index.
// Sheet cũ "data" không bị thay đổi. Triển khai phiên bản web app mới.
const SHEET_NAME = 'data_v31';
const COLLECTOR_VERSION = '3.1';
const ID_COLUMN = 789; // 1-based: label, writer, pixel0..783, time, strokes, sample_id

function doGet() {
  return HtmlService.createHtmlOutputFromFile('index').setTitle('Viết số · 33 mẫu')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}
function headers_() {
  const h = ['label', 'writer'];
  for (let i = 0; i < 784; i++) h.push('pixel' + i);
  return h.concat(['time', 'strokes', 'sample_id', 'collector_version', 'writer_name', 'batch_id']);
}
function getSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  if (!ss) throw new Error('CONFIG: Apps Script phải gắn với Google Sheet.');
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) sh = ss.insertSheet(SHEET_NAME);
  const h = headers_();
  if (sh.getMaxColumns() < h.length) sh.insertColumnsAfter(sh.getMaxColumns(), h.length - sh.getMaxColumns());
  if (sh.getLastRow() === 0) { sh.getRange(1, 1, 1, h.length).setValues([h]); sh.setFrozenRows(1); }
  else {
    const actual = sh.getRange(1, 1, 1, h.length).getValues()[0];
    if (!h.every((v, i) => v === actual[i])) throw new Error('CONFIG: Header data_v31 không đúng; không ghi dữ liệu.');
  }
  return sh;
}
function writer_(value) {
  const w = String(value || '').normalize('NFC').trim().toLowerCase();
  if (!/^[a-z0-9][a-z0-9_-]{2,39}$/.test(w)) throw new Error('INVALID: Mã người viết cần 3–40 ký tự a-z, 0-9, _ hoặc -.');
  return w;
}
function progress_(sh, writer) {
  const n = sh.getLastRow() - 1;
  if (!n) return [];
  const pairs = sh.getRange(2, 1, n, 2).getValues();
  const ids = sh.getRange(2, ID_COLUMN, n, 1).getValues();
  return pairs.reduce((a, r, i) => { if (r[1] === writer) a.push({ sample_id: ids[i][0], label: Number(r[0]) }); return a; }, []);
}
function getWriterProgress(writer) {
  writer = writer_(writer);
  const lock = LockService.getScriptLock(); lock.waitLock(20000);
  try { return { version: COLLECTOR_VERSION, writer: writer, samples: progress_(getSheet_(), writer) }; }
  finally { lock.releaseLock(); }
}
function saveSample(s) {
  if (!s || s.collector_version !== COLLECTOR_VERSION) throw new Error('INVALID: Cần trang thu v3.1.');
  const writer = writer_(s.writer), label = Number(s.label);
  if (!Number.isInteger(label) || label < 10 || label > 20) throw new Error('INVALID: label');
  if (typeof s.sample_id !== 'string' || !/^[a-zA-Z0-9_-]{16,80}$/.test(s.sample_id)) throw new Error('INVALID: sample_id');
  if (!Array.isArray(s.px) || s.px.length !== 784 || !s.px.every(v => typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= 255) || !s.px.some(v => v > 0)) throw new Error('INVALID: px');
  const st = s.strokes;
  if (!st || !Array.isArray(st.c) || st.c.length !== 2 || st.c[0] !== 600 || st.c[1] !== 400 || !Array.isArray(st.s) || !st.s.length) throw new Error('INVALID: strokes');
  for (const line of st.s) {
    if (!line || !Number.isInteger(line.w) || line.w < 28 || line.w > 52 || !Array.isArray(line.p) || line.p.length < 2 || line.p.length % 2 || !line.p.every(Number.isFinite)) throw new Error('INVALID: stroke points');
  }
  const strokes = JSON.stringify(st);
  if (strokes.length > 45000) throw new Error('INVALID: Nét quá dài; giữ mẫu trong bản dự phòng để kiểm tra.');
  const name = String(s.writer_name || '').normalize('NFC').replace(/\s+/g, ' ').trim().slice(0, 80).replace(/^[=+\-@]+/, '');
  const lock = LockService.getScriptLock(); lock.waitLock(20000);
  try {
    const sh = getSheet_(), n = sh.getLastRow() - 1;
    if (n) {
      const hit = sh.getRange(2, ID_COLUMN, n, 1).createTextFinder(s.sample_id).matchEntireCell(true).useRegularExpression(false).findNext();
      if (hit) {
        const row = sh.getRange(hit.getRow(), 1, 1, headers_().length).getValues()[0];
        if (row[0] !== label || row[1] !== writer || row[787] !== strokes || !s.px.every((v, i) => row[i + 2] === v)) throw new Error('INVALID: sample_id trùng nhưng nội dung khác.');
        return { version: COLLECTOR_VERSION, sample_id: s.sample_id, writer: writer, label: label, duplicate: true };
      }
    }
    if (progress_(sh, writer).filter(r => r.label === label).length >= 3) throw new Error('QUOTA: Mã này đã đủ 3 mẫu cho số ' + label + '. Bấm Đồng bộ.');
    sh.appendRow([label, writer].concat(s.px, [new Date(), strokes, s.sample_id, COLLECTOR_VERSION, name, 'new_writers_v31']));
    SpreadsheetApp.flush();
    return { version: COLLECTOR_VERSION, sample_id: s.sample_id, writer: writer, label: label, duplicate: false };
  } finally { lock.releaseLock(); }
}
