// Gắn vào Google Sheet: Tiện ích mở rộng > Apps Script, dán toàn bộ file này vào Code.gs.
// Tạo thêm một file HTML tên "index" (dán nội dung Index.html vào đó).

const SHEET_NAME = 'data';

function doGet() {
  return HtmlService.createHtmlOutputFromFile('index')
    .setTitle('Viết số')
    .addMetaTag('viewport', 'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');
}

function getSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) sh = ss.insertSheet(SHEET_NAME);
  if (sh.getLastRow() === 0) {
    const head = ['label', 'writer'];
    for (let i = 0; i < 784; i++) head.push('pixel' + i);
    head.push('time', 'strokes');
    sh.appendRow(head);
    sh.setFrozenRows(1);
  }
  return sh;
}

// Được gọi từ trang web mỗi lần bấm "Lưu mẫu": thêm đúng 1 dòng vào sheet "data".
function saveSample(s) {
  if (!s || !Array.isArray(s.px) || s.px.length !== 784) throw new Error('INVALID: px');
  const label = Number(s.label);
  if (!(label >= 10 && label <= 20)) throw new Error('INVALID: label');

  const writer = String(s.writer || '').replace(/^[=+\-@\s]+/, '').slice(0, 40);
  const px = s.px.map(function (v) { return Math.max(0, Math.min(255, Math.round(Number(v) || 0))); });
  let strokes = JSON.stringify(s.strokes || {});
  if (strokes.length > 45000) strokes = '';

  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const sh = getSheet_();
    const row = [label, writer].concat(px);
    row.push(new Date(), strokes);
    sh.appendRow(row);
    return sh.getLastRow() - 1;
  } finally {
    lock.releaseLock();
  }
}
