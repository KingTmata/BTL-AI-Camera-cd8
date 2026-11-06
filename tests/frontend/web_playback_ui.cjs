const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const html=fs.readFileSync(path.join(__dirname,'../../src/classroom_ai/interfaces/web.html'),'utf8');

function setup(enabled=true){
  const start=html.indexOf('// Playback diagnostics'),end=html.indexOf('// End playback diagnostics');
  assert.ok(start>=0&&end>start,'playback diagnostics must be available for an actual browser measurement');
  let now=100,dropped=5,total=10;
  const video={dataset:{},getVideoPlaybackQuality(){return {droppedVideoFrames:dropped,totalVideoFrames:total};}};
  const context=vm.createContext({video,location:{search:enabled?'?diagnostics=1':''},URLSearchParams,performance:{now:()=>now},JSON,Number,Math});
  vm.runInContext(html.slice(start,end),context);
  return {video,run:code=>vm.runInContext(code,context),time(value){now=value;},quality(d,t){dropped=d;total=t;}};
}

test('playback diagnostics measure native counter deltas and presented-frame gaps without changing media',()=>{
  const s=setup();s.run('startPlaybackDiagnostics()');
  s.run('recordPlaybackFrame(100,{mediaTime:0,presentedFrames:12,expectedDisplayTime:102,processingDuration:.003},.5,true)');
  s.run('recordPlaybackFrame(134,{mediaTime:.066,presentedFrames:14,expectedDisplayTime:135,processingDuration:.004},.8,false)');
  s.time(180);s.quality(7,14);s.run('finishPlaybackDiagnostics()');
  const report=JSON.parse(s.video.dataset.playbackStats);
  assert.equal(report.native.dropped,2);assert.equal(report.native.total,4);
  assert.equal(report.callbacks,2);assert.equal(report.presented_gaps,1);
  assert.equal(report.cache_misses,1);assert.equal(report.samples[0].decode_ms,3);
  assert.equal(report.samples[0].deadline_margin_ms,1.5);
  assert.equal(report.wall_ms,80);
  assert.equal(s.video.currentTime,undefined,'measurement must not seek or alter playback');
});

test('diagnostics bound their samples and report unavailable native counters honestly',()=>{
  const s=setup();delete s.video.getVideoPlaybackQuality;s.run('startPlaybackDiagnostics()');
  s.run('for(let i=0;i<900;i++)recordPlaybackFrame(i*33,{mediaTime:i/30,presentedFrames:i},.2,true)');
  s.run('finishPlaybackDiagnostics()');
  const report=JSON.parse(s.video.dataset.playbackStats);
  assert.equal(report.callbacks,900);assert.equal(report.samples.length,600);
  assert.equal(report.native,null);assert.equal(report.samples[0].deadline_margin_ms,null);
});

test('normal playback keeps diagnostics disabled and allocates no samples',()=>{
  const s=setup(false);s.run('startPlaybackDiagnostics();recordPlaybackFrame(0,{},1,false);finishPlaybackDiagnostics()');
  assert.equal(s.video.dataset.playbackStats,undefined);
  assert.equal(s.run('playbackDiagnostics.samples.length'),0);
});
