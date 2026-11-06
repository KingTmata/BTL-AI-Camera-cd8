// Exercise production inline overlay functions with a recording canvas context.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');
function setup(){
 const calls=[],nodes=new Map(),writes={width:0,height:0};
 const ctx=new Proxy({}, {get:(_,name)=> (...args)=>{calls.push([name,...args]);return name==='measureText'?{width:100}:undefined;},set:()=>true});
 const get=id=>{if(!nodes.has(id))nodes.set(id,{checked:true,dataset:{},style:{},width:1920,height:1080,replaceChildren(){},removeAttribute(name){delete this[name];},getContext(){return ctx;}});return nodes.get(id);};
 const canvas=get('overlay');for(const name of ['width','height']){let value=canvas[name];Object.defineProperty(canvas,name,{get(){return value;},set(next){writes[name]++;value=next;}});}
 Object.assign(get('video'),{videoWidth:1920,videoHeight:1080,clientWidth:720,clientHeight:405});
 Object.assign(get('stage'),{clientWidth:1200,clientHeight:675});
 get('showRaw').checked=false;
 const observations=new Map();
 class ResizeObserver{constructor(callback){this.callback=callback;}observe(target){observations.set(target,this.callback);}}
 const context=vm.createContext({document:{getElementById:get},window:{devicePixelRatio:2},ResizeObserver,Headers,URLSearchParams,AbortController,TextDecoder,Uint8Array,Float64Array,DataView});
 vm.runInContext(html.split('<script>')[1].split('if(videoFrameAPI){')[0],context);
 const run=code=>vm.runInContext(code,context);
 run(`job={state:'complete',preview_verified:true};source={width:1920,height:1080};config={pose_edges13:[[0,1],[1,2]]};playbackDiagnostics.enabled=true;playbackDiagnostics.active=true;`);
 return {run,get,calls,writes,observations};
}
test('overlay backing follows displayed area with source coordinates and does not reset unchanged dimensions',()=>{
 const s=setup();s.run('fitOverlay()');
 assert.equal(s.get('overlay').width,960);assert.equal(s.get('overlay').height,540);
 assert.equal(s.get('overlay').style.width,'720px');assert.equal(s.get('overlay').style.height,'405px');
 const first={...s.writes};s.run('fitOverlay()');assert.deepEqual(s.writes,first);
 assert.equal(s.run('playbackDiagnostics.resizes'),1);
 assert.equal(s.run('playbackDiagnostics.backing_writes'),2);
 s.run(`currentFrame={timestamp_seconds:1,segment_id:0,display_tracks:[{bbox_xyxy:[100,100,200,200],keypoints13:[[100,100,1],[200,200,1]]}]};currentFrameIndex=30;presentedMediaTime=1;drawOverlay();`);
 assert.ok(s.calls.some(c=>c[0]==='setTransform'&&c[1]===.5&&c[4]===.5));
 assert.equal(s.get('overlay').dataset.frameIndex,'30');assert.equal(s.get('overlay').dataset.mediaTime,'1');
});
test('normal pose uses display joints and batches edges and circles into one stroke/fill per person',()=>{
 const s=setup();s.get('showBoxes').checked=false;s.get('showIDs').checked=false;
 s.run(`currentFrame={timestamp_seconds:0,segment_id:0,display_tracks:[{bbox_xyxy:[0,0,300,300],keypoints13:[[10,10,1],[20,20,1],[30,30,1]],display_keypoints13:[[100,100,1],[200,200,1],[300,300,1]]}]};currentFrameIndex=0;drawOverlay();`);
 assert.equal(s.calls.filter(c=>c[0]==='stroke').length,1);assert.equal(s.calls.filter(c=>c[0]==='fill').length,1);
 assert.deepEqual(s.calls.filter(c=>c[0]==='arc').map(c=>c[1]),[100,200,300]);
 assert.equal(s.calls.filter(c=>c[0]==='moveTo').length,5,'each circle starts its own subpath without connecting filled disks');
});
test('tracked frame with no confirmed display rows stays empty; raw toggle explicitly selects detections',()=>{
 const s=setup();s.run(`currentFrame={timestamp_seconds:0,segment_id:0,display_tracks:[],detections:[{bbox_xyxy:[0,0,10,10],keypoints13:[[5,5,1]],display_keypoints13:[[8,8,1]]}]};drawOverlay();`);
 assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,0);
 s.get('showRaw').checked=true;s.run('drawOverlay()');
 assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,1);
 assert.equal(s.calls.filter(c=>c[0]==='arc').at(-1)[1],5);
});
test('untracked frame falls back to detections and retains pose confidence filtering',()=>{
 const s=setup();s.run(`currentFrame={timestamp_seconds:0,segment_id:null,display_tracks:[],detections:[{bbox_xyxy:[0,0,10,10],keypoints13:[[1,1,.05],[2,2,.049],[3,3,.2]]}]};drawOverlay();`);
 assert.equal(s.calls.filter(c=>c[0]==='strokeRect').length,1);
 assert.deepEqual(s.calls.filter(c=>c[0]==='arc').map(c=>c[1]),[1,3]);
 assert.equal(s.calls.filter(c=>c[0]==='lineTo').length,0);
});

test('video-height-only layout changes refit the canvas while the stage stays unchanged',()=>{
 const s=setup();
 const observerSetup=html.split(/\r?\n/).find(l=>l.includes(".observe($('stage'))"));
 s.run(observerSetup.split(';window.addEventListener')[0]);
 const refit=s.observations.get(s.get('video'));
 assert.equal(typeof refit,'function','actual video box must be observed, including max-height changes');
 s.run('fitOverlay()');
 s.get('video').clientHeight=300;refit();
 assert.equal(s.get('overlay').style.height,'300px');
 assert.equal(s.get('overlay').style.width,`${1920*300/1080}px`);
});

test('starting another job clears the entire transformed canvas before new media loads',()=>{
 const s=setup();Object.assign(s.get('video'),{pause(){},load(){},removeAttribute(){}});
 s.run(`fitOverlay();currentFrame={timestamp_seconds:0,segment_id:0,display_tracks:[{bbox_xyxy:[0,0,100,100]}]};drawOverlay();`);
 s.calls.length=0;s.run('clearViewer()');
 assert.equal(s.calls[0][0],'resetTransform');
 assert.deepEqual(s.calls[1],['clearRect',0,0,960,540]);
 assert.equal(s.get('overlay').dataset.frameIndex,undefined);
});
function actionCanvas(status='active'){
 const s=setup();s.get('showBoxes').checked=false;s.get('showPose').checked=false;s.get('showActions').checked=true;
 s.run(`job.processing_mode='replay';liveOverlayReady=()=>true;currentFrame={frame_index:30,sequence:5,timestamp_seconds:1,segment_id:2,tracks:[{track_id:7,confirmed:true,state:'observed',detection_index:0}],detections:[{bbox_xyxy:[10,30,80,150]}],display_tracks:[{track_id:7,confirmed:true,state:'observed',bbox_xyxy:[10,30,80,150]}],action_people:[{segment_id:2,track_id:7,status:'${status}',label:${['active','candidate','release_pending'].includes(status)?"'raise_hand'":'null'},origin:'rule',review:'draft',calibrated:false,rule:'coco17_wrist_above_nose_v1',observed_through_s:1}]};currentFrameIndex=30;`);return s;
}
test('current draft action text appears beside its exact ID independently of boxes and Pose',()=>{
 const s=actionCanvas();s.get('showIDs').checked=false;s.run('drawOverlay()');
 assert.match(s.calls.find(c=>c[0]==='fillText')[1],/ID 7.*Giơ tay · nháp/);
 assert.equal(s.calls.filter(c=>c[0]==='strokeRect'||c[0]==='arc').length,0);
 s.calls.length=0;s.get('showActions').checked=false;s.run('drawOverlay()');assert.equal(s.calls.filter(c=>c[0]==='fillText').length,0);
});
test('candidate, release, unsupported and no-suggestion states have distinct non-GT captions',()=>{
 for(const [state,text] of [['candidate','Giơ tay? · đang xác nhận'],['release_pending','Giơ tay · chờ kết thúc'],['unsupported','Chưa rõ hành động'],['no_suggestion','Chưa có gợi ý']]){
  const s=actionCanvas(state);s.run('drawOverlay()');assert.ok(s.calls.some(c=>c[0]==='fillText'&&c[1].includes(text)),state);
 }
});
test('raw detections, interpolated/unconfirmed/unmatched people and stale scene/time never inherit action text',()=>{
 for(const mutation of ["$('showRaw').checked=true", "currentFrame.display_tracks[0].state='interpolated'",'currentFrame.tracks[0].confirmed=false',"currentFrame.tracks[0].state='interpolated'",'currentFrame.action_people[0].segment_id=3','currentFrame.action_people[0].track_id=8','currentFrame.action_people[0].observed_through_s=.9',"currentFrame.action_people[0].origin='human'",'currentFrame.action_people[0].calibrated=true',"currentFrame.action_people[0].label='sitting'",'currentFrame.action_people.push({...currentFrame.action_people[0]})']){
  const s=actionCanvas();s.run(mutation+';drawOverlay()');assert.ok(!s.calls.some(c=>c[0]==='fillText'&&/Giơ tay|Chưa rõ hành động|Chưa có gợi ý/.test(c[1])),mutation);
 }
});
test('historical closed events never become a current action caption',()=>{
 const s=actionCanvas();s.run(`delete currentFrame.action_people;currentFrame.action_events=[{event:{label:'raise_hand',track_id:7,start_s:0,end_s:1,origin:'rule',review:'draft'}}];drawOverlay();`);
 assert.ok(s.calls.some(c=>c[0]==='fillText'&&c[1].includes('ID 7')));assert.ok(!s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Giơ tay')));
});
test('long ID action caption stays within the source canvas at the right edge',()=>{
 const s=actionCanvas('candidate');s.run(`currentFrame.display_tracks[0].bbox_xyxy=[1910,30,1919,150];drawOverlay();`);
 const background=s.calls.find(c=>c[0]==='fillRect'),text=s.calls.find(c=>c[0]==='fillText');assert.ok(background[1]+background[3]<=1920);assert.ok(text[2]<1910);
});
test('basic current-person actions show simultaneous draft behavior with completed past-tense captions',()=>{
 const s=actionCanvas();s.run(`currentFrame.action_people[0]={segment_id:2,track_id:7,observed_through_s:1,actions:[{label:'raise_hand',status:'active',rule:'coco17_wrist_above_nose_v1'},{label:'stand_up',status:'completed',rule:'coco17_sit_to_stand_v1',confirmed_at_s:1},{label:'move_to_board',status:'completed',rule:'coco17_floor_roi_arrival_v1',confirmed_at_s:1}].map(a=>({...a,origin:'rule',review:'draft',calibrated:false}))};drawOverlay();`);
 const texts=s.calls.filter(c=>c[0]==='fillText').map(c=>c[1]).join(' ');assert.match(texts,/Giơ tay · nháp/);assert.match(texts,/Vừa đứng lên · nháp/);assert.match(texts,/Đến vùng bảng · nháp/);assert.doesNotMatch(texts,/Đang đứng|Viết bảng/);
 assert.ok(s.calls.filter(c=>c[0]==='fillText').length<=2,'current-person caption wraps at most two lines');
});
test('basic status gates never turn constant labels into positive actions or display an unconfigured board',()=>{
 const s=actionCanvas();s.run(`currentFrame.action_people[0]={segment_id:2,track_id:7,observed_through_s:1,actions:[{label:'raise_hand',status:'unsupported',rule:'coco17_wrist_above_nose_v1'},{label:'stand_up',status:'no_suggestion',rule:'coco17_sit_to_stand_v1'},{label:'move_to_board',status:'not_configured',rule:'coco17_floor_roi_arrival_v1'}].map(a=>({...a,origin:'rule',review:'draft',calibrated:false}))};drawOverlay();`);
 const texts=s.calls.filter(c=>c[0]==='fillText').map(c=>c[1]).join(' ');assert.match(texts,/Chưa rõ hành động/);assert.doesNotMatch(texts,/Giơ tay · nháp|Vừa đứng|Đến vùng|not_configured/);
});
test('basic candidate captions remain questions and unrelated rule provenance is omitted',()=>{
 const s=actionCanvas();s.run(`currentFrame.action_people[0]={segment_id:2,track_id:7,observed_through_s:1,actions:[{label:'stand_up',status:'candidate',rule:'coco17_sit_to_stand_v1'},{label:'move_to_board',status:'candidate',rule:'coco17_floor_roi_arrival_v1'},{label:'write_board',status:'active',rule:'invented'}].map(a=>({...a,origin:'rule',review:'draft',calibrated:false}))};drawOverlay();`);
 const texts=s.calls.filter(c=>c[0]==='fillText').map(c=>c[1]).join(' ');assert.match(texts,/Đang đứng lên\? · nháp/);assert.match(texts,/Đang tiến về bảng\? · nháp/);assert.doesNotMatch(texts,/Viết bảng|invented/);
});

test('lowered-hand caption requires explicit supported Pose observation, never no-suggestion alone',()=>{
 const s=actionCanvas('no_suggestion');s.run(`Object.assign(currentFrame.action_people[0],{pose_observation:'not_raised',pose_observation_reason:'both_wrists_lowered',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',reason:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false}});drawOverlay();`);
 assert.ok(s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Tay đang hạ · Pose nháp')));
 for(const mutation of ["delete currentFrame.action_people[0].pose_observation","currentFrame.action_people[0].pose_observation='unknown'","currentFrame.action_people[0].pose_quality.status='unsupported'","currentFrame.action_people[0].pose_quality.gate_version='other'","currentFrame.action_people[0].pose_observation_scope='display_estimate'","currentFrame.action_people[0].pose_observation_reason='ambiguous_geometry'"]){
  const t=actionCanvas('no_suggestion');t.run(`Object.assign(currentFrame.action_people[0],{pose_observation:'not_raised',pose_observation_reason:'both_wrists_lowered',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false}});${mutation};drawOverlay();`);assert.ok(!t.calls.some(c=>c[0]==='fillText'&&c[1].includes('Tay đang hạ')),mutation);
 }
});
test('lowered Pose geometry does not replace the temporal release-pending event state',()=>{
 const s=actionCanvas('release_pending');s.run(`Object.assign(currentFrame.action_people[0],{pose_observation:'not_raised',pose_observation_reason:'both_wrists_lowered',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false}});drawOverlay();`);
 const text=s.calls.filter(c=>c[0]==='fillText').map(c=>c[1]).join(' ');assert.match(text,/Giơ tay · chờ kết thúc/);assert.match(text,/Tay đang hạ · Pose nháp/);
});
test('malformed raise state cannot display a supported negative Pose caption',()=>{
 const s=actionCanvas('no_suggestion');s.run(`Object.assign(currentFrame.action_people[0],{status:'invented',pose_observation:'not_raised',pose_observation_reason:'both_wrists_lowered',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false}});drawOverlay();`);
 assert.ok(!s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Tay đang hạ')));
});
test('shoulder cue has an exact whitelist and the same Pose quality and provenance gates',()=>{
 for(const status of ['active','no_suggestion']){
  const s=actionCanvas(status);s.run(`Object.assign(currentFrame.action_people[0],{rule:'coco17_shoulder_any_arm_v1',pose_observation:'${status==='active'?'raised':'not_raised'}',pose_observation_reason:'${status==='active'?'raised_pose':'both_wrists_lowered'}',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false,supported_arms:${status==='active'?"['right']":"['left','right']"}}});drawOverlay();`);
  assert.ok(s.calls.some(c=>c[0]==='fillText'&&c[1].includes(status==='active'?'Giơ tay · nháp':'Tay đang hạ · Pose nháp')));
 }
 for(const mutation of ["currentFrame.action_people[0].rule='coco17_shoulder_fake_v1'","currentFrame.action_people[0].pose_quality.status='unsupported'","currentFrame.action_people[0].pose_observation_scope='display_estimate'","currentFrame.action_people[0].origin='human'"]){
  const s=actionCanvas();s.run(`Object.assign(currentFrame.action_people[0],{rule:'coco17_shoulder_any_arm_v1',pose_observation:'raised',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false,supported_arms:['right']}});${mutation};drawOverlay();`);assert.ok(!s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Giơ tay · nháp')),mutation);
 }
});
test('unsupported shoulder evidence stays unknown and one arm cannot establish lowered-hand negative',()=>{
 const s=actionCanvas('unsupported');s.run(`Object.assign(currentFrame.action_people[0],{rule:'coco17_shoulder_any_arm_v1',pose_observation:'unknown',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'unsupported',gate_version:'causal_pose_quality_v1',temporal_guard:false,supported_arms:[]}});drawOverlay();`);assert.ok(s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Chưa rõ hành động')));
 const one=actionCanvas('no_suggestion');one.run(`Object.assign(currentFrame.action_people[0],{rule:'coco17_shoulder_any_arm_v1',pose_observation:'not_raised',pose_observation_reason:'both_wrists_lowered',pose_observation_scope:'uncalibrated_raw_pose_geometry',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false,supported_arms:['right']}});drawOverlay();`);assert.ok(!one.calls.some(c=>c[0]==='fillText'&&c[1].includes('Tay đang hạ')));
});
function relationCanvas(){
 const s=actionCanvas();s.run(`currentFrame.action_people[0]={segment_id:2,track_id:7,observed_through_s:1,actions:[{label:'raise_hand',status:'no_suggestion',rule:'coco17_wrist_above_nose_v1',pose_quality:{status:'supported',gate_version:'causal_pose_quality_v1',temporal_guard:false}},{label:'move_to_board',status:'completed',start_s:.5,confirmed_at_s:.9,rule:'coco17_floor_roi_arrival_v1'}].map(a=>({...a,origin:'rule',review:'draft',calibrated:false})),context_relations:[{relation:'raise_hand_then_move_to_board',status:'observed_sequence',rule:'raise_hand_then_move_to_board_v1',origin:'rule',review:'draft',calibrated:false,raise_start_s:0,raise_confirmed_at_s:.2,raise_observed_through_s:.8,board_start_s:.5,board_confirmed_at_s:.9,max_separation_s:60,continuity:'same_scene_track_continuous_supported_pose'}]};`);return s;
}
test('raise-then-board relation is separate draft context and permits overlapping hand observation',()=>{
 const s=relationCanvas();s.run('drawOverlay()');const text=s.calls.filter(c=>c[0]==='fillText').map(c=>c[1]).join(' ');assert.match(text,/Đến vùng bảng · nháp/);assert.match(text,/Liên hệ: Đã giơ tay → đến vùng bảng · nháp/);
});
test('relation provenance, chronology, current board and uncertain association fail closed',()=>{
 for(const mutation of ["currentFrame.action_people[0].context_relations[0].origin='human'","currentFrame.action_people[0].context_relations[0].rule='invented'","currentFrame.action_people[0].context_relations[0].raise_confirmed_at_s=.6","currentFrame.action_people[0].context_relations[0].board_confirmed_at_s=2","currentFrame.action_people[0].context_relations[0].max_separation_s=61","currentFrame.action_people[0].context_relations[0].continuity='assumed'","currentFrame.action_people[0].actions[1].status='candidate'","currentFrame.action_people[0].actions[0].pose_quality.status='unsupported'","currentFrame.tracks[0].identity_uncertain=true","currentFrame.action_people[0].context_relations.push({...currentFrame.action_people[0].context_relations[0]})"]){
  const s=relationCanvas();s.run(mutation+';drawOverlay()');assert.ok(!s.calls.some(c=>c[0]==='fillText'&&c[1].includes('Đã giơ tay →')),mutation);
 }
});
