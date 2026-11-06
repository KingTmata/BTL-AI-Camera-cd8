const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict'),{test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
 const nodes=new Map();
 const node=()=>({value:'',checked:false,disabled:false,hidden:false,textContent:'',children:[],dataset:{},style:{},files:[],append(...xs){this.children.push(...xs);},replaceChildren(...xs){this.children=xs;},addEventListener(){},removeAttribute(name){delete this[name];},querySelectorAll(){return [];},querySelector(){return node();},getContext(){return {clearRect(){}};},reportValidity(){return true;}});
 const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
 const context=vm.createContext({document:{getElementById:get,createElement:node},window:{},Headers,URL,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView,performance:{now:()=>1000},setTimeout:()=>1,clearTimeout(){}});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context),text=n=>[n.textContent,...n.children.map(text)].join(' ');
 run(`config={models:[]};job=null;$('processingMode').value='full';$('tracking').value='bytetrack';$('trackingProfile').value='baseline';$('task').value='pose';$('confidence').value='.05';$('source').value='local.mov';$('model').value='pose';$('imgsz').value='1280';$('device').value='cpu';$('nms').value='';`);
 return {get,run,text};
}
test('tracking candidate is explicit opt-in; baseline retains null threshold',()=>{
 const s=setup();assert.match(html,/<select id="trackingProfile"><option value="baseline"/);
 assert.equal(s.run('submissionTrackingOptions().new_track_thresh'),null);
 s.get('trackingProfile').value='new_010';assert.equal(s.run('submissionTrackingOptions().new_track_thresh'),.1);
 s.get('confidence').value='.25';assert.equal(s.run('submissionTrackingOptions().new_track_thresh'),.25);
 s.get('tracking').value='off';assert.equal(s.run('submissionTrackingOptions().new_track_thresh'),null);
});
test('tracking candidate controls are disabled for prep, tracking off and active work',()=>{
 const s=setup();s.run('updateProcessingOptions()');assert.equal(s.get('trackingProfile').disabled,false);
 for(const mutation of ["$('processingMode').value='annotation_prep'","$('tracking').value='off'","job={state:'processing'}","submitting=true","job={video_export:{state:'running'}}"]){
  s.run("$('processingMode').value='full';$('tracking').value='bytetrack';job=null;submitting=false;"+mutation+';updateProcessingOptions()');assert.equal(s.get('trackingProfile').disabled,true,mutation);
 }
});
test('actual job submission forwards candidate threshold without changing detector confidence',async()=>{
 const s=setup();s.get('trackingProfile').value='new_010';s.run(`let sent=[];post=async(path,body)=>{sent.push({path,body});return {id:'${'a'.repeat(32)}'};};clearViewer=()=>{};renderAnnotations=()=>{};renderArtifacts=()=>{};renderJob=()=>{};poll=()=>{};storageSet=()=>{};replaceCurrentJobURL=()=>{};updateControls=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('sent[0].body.new_track_thresh'),.1);assert.equal(s.run('sent[0].body.confidence'),.05);assert.equal(s.run('sent[0].path'),'/api/jobs');
});
test('assistant annotation is labeled as assistant draft, including coverage',()=>{
 const s=setup();s.run(`selectedActor=()=>null;renderTimeline=()=>{};events=[{segment_id:0,track_id:'human:x',label:'raise_hand',start_s:0,end_s:1,origin:'assistant',review:'draft'}];coverage=[{segment_id:0,track_id:'human:x',label:'raise_hand',start_s:0,end_s:1,origin:'assistant',review:'draft'}];$('onlyActor').checked=false;renderAnnotations();`);
 const text=s.text(s.get('annotationRows'));assert.equal((text.match(/Trợ lý · nháp/g)||[]).length,2);assert.doesNotMatch(text,/Người gán nhãn|Gợi ý quy tắc/);
});
test('hand cue preserves legacy default and rejects unknown choices',()=>{
 const s=setup();assert.match(html,/<select id="handCue"><option value="nose_both_arms_v1"/);
 assert.equal(s.run('selectedHandCue()'),'nose_both_arms_v1');
 s.get('handCue').value='shoulder_any_arm_v1';assert.equal(s.run('selectedHandCue()'),'shoulder_any_arm_v1');
 s.get('handCue').value='invented';assert.throws(()=>s.run('selectedHandCue()'),/cue|tiêu chí/i);
});
test('job and replay requests snapshot the explicit hand cue and prep still omits AI configuration',async()=>{
 const s=setup();s.get('handCue').value='shoulder_any_arm_v1';s.run(`let sent=[];post=async(path,body)=>{sent.push({path,body});return {id:'${'a'.repeat(32)}',processing_mode:body.processing_mode};};clearViewer=()=>{};renderAnnotations=()=>{};renderArtifacts=()=>{};renderJob=()=>{};poll=()=>{};storageSet=()=>{};replaceCurrentJobURL=()=>{};updateControls=()=>{};`);
 await s.run('startJob({preventDefault(){}})');assert.equal(s.run('sent[0].body.hand_cue'),'shoulder_any_arm_v1');
 s.run("job=null;$('processingMode').value='replay';$('actionRules').value='raise_hand';");await s.run('startJob({preventDefault(){}})');assert.equal(s.run('sent[1].body.hand_cue'),'shoulder_any_arm_v1');
 s.run("job=null;$('processingMode').value='annotation_prep';");await s.run('startJob({preventDefault(){}})');assert.deepEqual(JSON.parse(s.run('JSON.stringify(sent[2].body)')),{source:'local.mov'});
});
test('proposal request pins hand cue; changing selector never rewrites an existing annotation draft',async()=>{
 const s=setup();s.get('handCue').value='shoulder_any_arm_v1';s.run(`job={id:'one',state:'complete'};events=[{label:'HUMAN',origin:'human'}];coverage=[{label:'COVERAGE'}];dirty=true;let sentProposal;post=async(_path,body)=>{sentProposal=body;return {hand_cue:'shoulder_any_arm_v1',events:[]};};updateControls=()=>{};prepareActionCaptions=()=>Promise.resolve();drawOverlay=()=>{};`);
 await s.run('proposals()');assert.equal(s.run('sentProposal.hand_cue'),'shoulder_any_arm_v1');
 s.get('handCue').value='nose_both_arms_v1';await s.run('handCueChanged()');assert.equal(s.run('events[0].label'),'HUMAN');assert.equal(s.run('coverage[0].label'),'COVERAGE');assert.equal(s.run('dirty'),true);
});
test('late proposals for a different selected cue cannot append to the draft',async()=>{
 const s=setup();s.get('handCue').value='shoulder_any_arm_v1';s.run(`job={id:'one',state:'complete'};events=[];let resolveProposal;post=()=>new Promise(resolve=>resolveProposal=resolve);updateControls=()=>{};`);
 const pending=s.run('proposals()');s.get('handCue').value='nose_both_arms_v1';s.run(`resolveProposal({hand_cue:'shoulder_any_arm_v1',events:[{label:'raise_hand'}]});`);await pending;assert.equal(s.run('events.length'),0);
});
