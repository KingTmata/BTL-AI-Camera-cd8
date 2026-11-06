const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
  const nodes=new Map();const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',disabled:false,textContent:'',replaceChildren(){}});return nodes.get(id);};
  const context=vm.createContext({document:{getElementById:get}});
  vm.runInContext(`const $=id=>document.getElementById(id);let config=null;function updateControls(){}\n`+['updateDeviceOptions','modelsForTask'].map(n=>html.split(/\r?\n/).find(l=>l.startsWith('function '+n+'('))).join('\n'),context);
  return {get,run:code=>vm.runInContext(code,context)};
}
test('Intel option is available for Pose and resets to CPU when switching to Detect',()=>{
  const s=setup();s.get('task').value='pose';s.get('device').value='intel:gpu';s.run('updateDeviceOptions()');
  assert.equal(s.get('intelGPU').disabled,false);assert.equal(s.get('device').value,'intel:gpu');
  s.get('task').value='detect';s.run('updateDeviceOptions()');
  assert.equal(s.get('intelGPU').disabled,true);assert.equal(s.get('device').value,'cpu');
});
test('changing task during cold config loading remains usable without a JavaScript exception',()=>{
  const s=setup();s.get('task').value='pose';
  assert.doesNotThrow(()=>s.run('modelsForTask()'));
  assert.equal(s.get('intelGPU').disabled,false);
});
