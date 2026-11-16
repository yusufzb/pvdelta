// Pure JavaScript unit test with a minimal DOM stub; not a browser/render test.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const template = fs.readFileSync(path.join(__dirname, '../src/pvdelta/region_annotation_editor.html'), 'utf8');
const frames = [0, 1].map(i => ({case_id: `C${i}`, source_sha256: String(i).repeat(64),
  native_temperature_sha256: 'a'.repeat(64), display_png_sha256: 'b'.repeat(64),
  native_width: 80, native_height: 60, image_data_url: 'synthetic',
  rgb_source_file: `paired-${i}.jpg`, rgb_data_url: 'synthetic'}));
const pack = {pack_token: 'synthetic-pack', frames};
const elements = new Map();
const context = new Proxy({}, {get: () => () => {}});
function element(id) {
  if (!elements.has(id)) elements.set(id, {value: '', checked: false, textContent: '',
    style: {}, appendChild() {}, getContext: () => context,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 160, height: 120}),
    setPointerCapture() {}});
  return elements.get(id);
}
element('mode').value = 'regions';
const sandbox = {document: {getElementById: element, createElement: () => ({})},
  Image: class {set src(value) {if (this.onload) this.onload();}}, console};
vm.createContext(sandbox);
const code = template.split('<script>')[1].split('</script>')[0]
  .replace('__PVDELTA_PACK__', JSON.stringify(pack));
vm.runInContext(code, sandbox);
const read = expression => vm.runInContext(expression, sandbox);
function click(x, y) {element('canvas').onpointerdown({preventDefault() {}, clientX: x*2, clientY: y*2});}
click(10, 10);
element('next').onclick();
assert.equal(read('index'), 0, 'pending drawing must not vanish on Next');
click(20, 10); click(20, 20); click(10, 20);
assert.equal(read('work().regions.length'), 0, 'four clicks no longer auto-close');
element('finish').onclick();
assert.equal(read('work().regions.length'), 1);
assert.deepEqual(JSON.parse(read('JSON.stringify(work().regions[0].polygon_native_xy)')),
  [[10,10], [20,10], [20,20], [10,20]], 'display scale must not alter native coordinates');
element('complete').checked = true;
element('complete').onchange();
assert.equal(read('work().exhaustive'), false, 'name is mandatory');
element('reviewer').value = 'synthetic test';
element('complete').checked = true;
element('complete').onchange();
assert.equal(read('work().exhaustive'), true);
element('next').onclick();
assert.equal(read('index'), 1);
assert.equal(read('work().regions.length'), 0, 'next empty frame is unresolved');
assert.equal(read('work().exhaustive'), false);
element('previous').onclick();
assert.equal(read('work().regions.length'), 1, 'Previous must retain drawing');
assert.equal(read('work().exhaustive'), true);
const exported = JSON.parse(read('JSON.stringify(exportDocument())'));
assert.equal(exported.schema, 'pvdelta.thermal-region-work/v1');
assert.equal(exported.frames.length, 2);
assert.equal(exported.frames[0].regions[0].region_id, 'R0001');
element('undo').onclick();
assert.equal(read('work().exhaustive'), false, 'geometry edit invalidates completion');
assert.equal(read('work().regions.length'), 0);
for (const p of [[10,10],[30,10],[30,20],[20,20],[20,30],[10,30]]) click(...p);
click(10,10);
assert.equal(read('work().regions[0].polygon_native_xy.length'), 6, 'concave shape closes on first point');
assert.equal(read('simple([[0,0],[10,10],[0,10],[10,0]])'), false, 'crossed shape rejected');
assert.equal(read('simple([[0,0],[10,0],[10,10]])'), true, 'triangle supported');
const polygonExport = JSON.parse(read('JSON.stringify(exportDocument())'));
async function resumeChecks() {
  for (const draft of [exported, polygonExport]) {
    await element('import').onchange({target: {files: [{text: async () => JSON.stringify(draft)}], value: ''}});
    assert.match(element('error').textContent, /Draft resumed/);
    assert.equal(read('work().regions[0].polygon_native_xy.length'), draft.frames[0].regions[0].polygon_native_xy.length);
  }
  console.log('PASS: native coordinates, concave polygons, first-point closure, crossing rejection, navigation, old/new draft resume and completion');
}
resumeChecks().catch(error => {console.error(error); process.exitCode = 1;});
