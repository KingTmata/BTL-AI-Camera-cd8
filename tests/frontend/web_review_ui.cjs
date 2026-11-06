// Execute the real inline review functions with a minimal form DOM. No browser packages.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = fs.readFileSync(require('node:path').join(__dirname,'../../src/classroom_ai/interfaces/web.html'), 'utf8');
const names = ['reviewFormState','actorKey','selectedActor','sameActor','renderActors','actorChanged','interval','eventEvidence','addAnnotation','editAnnotation','frameAtTime','replayVerified','requestFrame','drawOverlay'];
function setup() {
  const nodes = new Map();
  function node() { return {value:'', checked:false, hidden:false,dataset:{},width:640,height:480,getContext(){return {clearRect(){}};}, children:[], append(child){this.children.push(child);if(this.children.length===1)this.value=child.value;},replaceChildren(){this.children=[];this.value='';},focus(){}}; }
  const context = vm.createContext({Number,JSON,Error,document:{createElement:node},nodes});
  context.$ = id => {if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  vm.runInContext(`let segments=[],events=[],coverage=[],edit=null,reviewFormBaseline=null,lastError='',persisted=0;
    let job=null,source=null,frameTimes=null,timeIndexError=null,currentFrame=null,currentFrameIndex=-1,requestedFrame=-1,presentedMediaTime=null,reviewReady=false,jobEpoch=0,dirty=false;
    let frameCache=new Map(); const video={readyState:0}; let requestedIndices=[],apiCalls=0;
    function requestFrameIndex(index){requestedIndices.push(index);} function renderArtifacts(){} function updateControls(){} function updateSaveState(){} function drawBoardROI(){} function drawHumanReferences(){} function cancelHumanReference(){} function renderHumanReferences(){} function isAnnotationSource(){return false;}
    function jobPath(path=''){return '/api/jobs/'+job.id+path;} function loadFrameTimes(){apiCalls++;throw Error('unexpected time-index request');}
    async function api(){apiCalls++;throw Error('unexpected API request');}
    function resetEdit(){edit=null;} function renderAnnotations(){} function clearError(){lastError='';}
    function error(message){lastError=message;} function persistDraft(){persisted++;}
    ${names.map(name=>html.split(/\r?\n/).find(line=>line.startsWith('function '+name+'('))).join('\n')}
    ${html.slice(html.indexOf('async function loadComplete('),html.indexOf('\nfunction renderArtifacts('))}`,context);
  return {context,run:code=>vm.runInContext(code,context)};
}
test('later contiguous extent is selectable and editing picks its containing extent',()=>{
  const {run}=setup();
  run(`segments=[{segment_id:0,track_id:1,start_s:0,end_s:1},{segment_id:0,track_id:1,start_s:2,end_s:3}];renderActors();`);
  assert.equal(run(`new Set($('actor').children.map(o=>o.value)).size`),2);
  run(`$('actor').value=$('actor').children[1].value;actorChanged();`);
  assert.equal(run(`selectedActor().start_s`),2);
  run(`events=[{segment_id:0,track_id:1,start_s:2.1,end_s:2.9,label:'raise_hand',origin:'rule',review:'draft',evidence:{}}];editAnnotation('event',0);`);
  assert.equal(run(`selectedActor().start_s`),2);
});
test('fractional endpoints survive default form and exact bound validation',()=>{
  const {run}=setup();
  run(`segments=[{segment_id:0,track_id:1,start_s:0,end_s:2/30}];renderActors();`);
  assert.equal(run(`interval().end_s`),2/30);
  run(`$('endTime').value='0.067';`);
  assert.throws(()=>run(`interval()`),/phạm vi/);
});
test('review preserves rule origin and censored boundary evidence',()=>{
  const {run}=setup();
  run(`segments=[{segment_id:0,track_id:1,start_s:0,end_s:4}];renderActors();events=[{segment_id:0,track_id:1,start_s:1,end_s:3,label:'raise_hand',origin:'rule',review:'draft',uncertain:false,evidence:{boundary_truncated:{start:true,end:false}}}];editAnnotation('event',0);$('review').value='reviewed';addAnnotation('event');`);
  assert.equal(run(`events[0].origin`),'rule');
  assert.equal(run(`events[0].review`),'reviewed');
  assert.equal(run(`events[0].evidence.boundary_truncated.start`),true);
});
test('partly known boundary pair is rejected before draft mutation',()=>{
  const {run}=setup();
  run(`segments=[{segment_id:0,track_id:1,start_s:0,end_s:4}];renderActors();$('label').value='raise_hand';$('startBoundary').value='truncated';$('endBoundary').value='unknown';addAnnotation('event');`);
  assert.equal(run(`events.length`),0);
  assert.equal(run(`persisted`),0);
  assert.match(run(`lastError`),/biên|đầu|cuối/);
  run(`events=[{segment_id:0,track_id:1,start_s:1,end_s:3,label:'raise_hand',origin:'rule',review:'draft',uncertain:false,evidence:{boundary_truncated:{start:true,end:false}}}];editAnnotation('event',0);$('endBoundary').value='unknown';addAnnotation('event');`);
  assert.equal(run(`events[0].review`),'draft');
  assert.equal(run(`events[0].evidence.boundary_truncated.start`),true);
  assert.equal(run(`persisted`),0);
});
test('microsecond-truncated browser presentation time picks the displayed frame',()=>{
  const {run}=setup();
  assert.equal(run(`frameAtTime([0,1/30,2/30,.1],.066666)`),2);
  assert.equal(run(`frameAtTime([0,1/30,2/30,.1],.066665)`),1);
  assert.equal(run(`frameAtTime([0,.000001,.000002,.1],.000000)`),0);
});
test('legacy completion leaves raw video available and blocks overlay/review API fetches',async()=>{
  const {run}=setup();
  run(`job={id:'legacy',state:'complete',preview_verified:false,preview_warning:'unverified legacy preview',artifacts:['preview.mp4']};`);
  await run(`loadComplete(0)`);
  assert.equal(run(`reviewReady`),false);
  assert.equal(run(`apiCalls`),0);
  assert.equal(run(`video.src`),'/api/jobs/legacy/artifacts/preview.mp4');
  assert.equal(run(`lastError`),'unverified legacy preview');
  run(`frameTimes=[0,1/30,2/30];requestFrame(.066666);`);
  assert.equal(run(`requestedIndices.length`),0);
});
test('canvas read-only diagnostics describe the actual requested/raster frame',()=>{
  const {run}=setup();
  run(`job={state:'complete',preview_verified:true};frameTimes=[0,1/30,2/30];requestFrame(.066666);currentFrameIndex=requestedIndices[0];currentFrame={frame_index:2,timestamp_seconds:2/30,detections:[],display_tracks:[]};drawOverlay();`);
  assert.equal(run(`$('overlay').dataset.mediaTime`),'0.066666');
  assert.equal(run(`$('overlay').dataset.frameIndex`),'2');
  assert.equal(run(`Number($('overlay').dataset.frameTimestamp)`),2/30);
  run(`currentFrame=null;drawOverlay();`);
  assert.equal(run(`Object.keys($('overlay').dataset).length`),0);
});
