"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const nodes = [
  {id:"video",title:"Video lớp học",short:"Nguồn quan sát",icon:"▶",x:2,y:2,status:"Đã có",input:"Video local từ camera lớp học. Ví dụ trong ảnh: 1920 × 1080, 25 FPS, HEVC trong MPEG-PS.",process:"Giữ bản nguồn; kiểm quyền sử dụng, góc quay, ánh sáng và độ rõ của những hành vi cần quan sát.",output:"Đường dẫn nguồn, ngữ cảnh buổi học và video gốc để đối chiếu.",note:"Mẫu MediaInfo chỉ cho biết thông số kỹ thuật, chưa chứng minh chất lượng nhận dạng hoặc quyền sử dụng dữ liệu."},
  {id:"probe",title:"Kiểm tra dữ liệu",short:"Container · codec · thời gian",icon:"⌕",x:37.3,y:2,status:"Đã có",input:"File video thật, không suy định dạng từ phần mở rộng tên file.",process:"ffprobe đọc stream; pipeline video kiểm timestamp và nguồn CFR. OpenCV giải mã thành frame trong RAM.",output:"Thông số nguồn và frame ảnh đúng trục thời gian cho bước phát hiện.",note:"Công cụ metadata trên web chỉ đọc thông số. Kiểm CFR toàn bộ và thử giải mã được pipeline video thực hiện riêng."},
  {id:"detection",title:"Phát hiện đối tượng",short:"YOLO · 8 lớp · lọc person",icon:"▣",x:72.6,y:2,status:"Đã có",input:"Frame ảnh sau giải mã, resize/padding theo imgsz; mặc định video hiện dùng 1280.",process:"YOLO dự đoán box, lớp và confidence cho 8 lớp: người, bàn, ghế, laptop, điện thoại, ba lô, sách, chai. Có thể bật person-only để chỉ giữ người; Conf/NMS chọn trước khi chạy.",output:"Detection gốc theo frame/timestamp trong detections.jsonl; box quy về kích thước nguồn.",note:"Confidence thấp có thể giữ detection yếu và tăng báo nhầm. Cần nhãn chuẩn của video lớp học để đo Recall trên nguồn này."},
  {id:"tracking",title:"Theo dõi diễn biến",short:"ByteTrack · ID cục bộ",icon:"↝",x:72.6,y:37,status:"Đã có · chỉ Full",input:"Detection person của các frame liên tiếp trong chế độ Full.",process:"ByteTrack không ReID, xác nhận hai hit liên tiếp, nhớ track khoảng 3 giây. EMA 0,65; chỉ nội suy khoảng ≤0,3 giây có hai đầu xác nhận cùng ID/cảnh.",output:"tracks_raw.jsonl và tracks.jsonl; tách observed/interpolated để kiểm tra minh chứng.",note:"ID cục bộ không phải danh tính sinh viên. Fast chưa tracking; che khuất dài hoặc đổi cảnh có thể làm đổi ID."},
  {id:"behavior",title:"Nhận dạng hành vi",short:"Chuỗi thời gian & ngữ cảnh",icon:"↟",x:37.3,y:37,status:"Đang nghiên cứu",planned:true,input:"Chuỗi hình ảnh hoặc tư thế, track trong khoảng quan sát và nhãn hành vi đã kiểm duyệt.",process:"Khảo sát phương pháp nhận dạng giơ tay, đứng lên, di chuyển lên bảng, viết bảng theo diễn biến và ngữ cảnh.",output:"Hướng tới sự kiện có nhãn, thời điểm bắt đầu/kết thúc và frame minh chứng.",note:"Chưa tích hợp bộ nhận dạng hành vi hoàn chỉnh. YOLO Pose, RTMPose, ST-GCN, PoseC3D, Jev mới là phương án khảo sát."},
  {id:"aggregation",title:"Tổng hợp theo thời gian",short:"Tần suất · thời lượng",icon:"≋",x:2,y:37,status:"Đang nghiên cứu",planned:true,input:"Sự kiện hành vi được xác nhận, thời lượng và ngữ cảnh của từng khoảng video.",process:"Liên kết sự kiện để hạn chế đếm lặp; tổng hợp cả lớp theo từng khoảng thời gian và toàn buổi. Nghiên cứu ngưỡng phân mức với giảng viên.",output:"Tần suất, thời lượng, diễn biến tương tác của lớp và tham chiếu minh chứng.",note:"Không yêu cầu hồ sơ/điểm từng sinh viên hoặc ID xuyên suốt buổi; không suy mức tương tác từ một hành động đơn lẻ."},
  {id:"report",title:"Báo cáo cho giảng viên",short:"Theo khoảng & cả buổi",icon:"↗",x:37.3,y:72,status:"Đầu ra hướng tới",planned:true,input:"Tổng hợp tương tác của lớp, tiêu chí đã kiểm định và các mốc video minh chứng.",process:"Trực quan hóa diễn biến, nêu độ tin cậy và giới hạn. Đối chiếu nhận định của giảng viên, hướng tới tích hợp AI-Native.",output:"Thông tin tham khảo giúp giảng viên theo dõi sự tham gia và điều chỉnh hoạt động giảng dạy.",note:"Báo cáo tương tác lớp là đầu ra nghiên cứu. Báo cáo hiện có trong pipeline chủ yếu là detection/tracking và số đo kỹ thuật."}
];

const mindBranches = [
  {id:"scope",title:"Mục tiêu & phạm vi",sub:"Lớp · khoảng thời gian · buổi học",col:1,row:1,input:"Video lớp học và yêu cầu hỗ trợ giảng viên đánh giá tương tác.",process:"Nghiên cứu hành vi quan sát được; phân biệt quan sát và diễn giải, tập trung tổng hợp cả lớp theo khoảng.",output:"Minh chứng để giảng viên tham khảo; không yêu cầu hồ sơ từng sinh viên.",note:"Tên đề tài và mục tiêu chính thức được đọc trực tiếp từ tasks/plan.md; phạm vi 22/10 được ghi trong project_memory.md.",status:"Phạm vi hiện hành"},
  {id:"data",title:"Dữ liệu đầu vào",sub:"Video · metadata · nhãn thời gian",col:3,row:1},
  {id:"models",title:"Mô hình & pipeline",sub:"Phát hiện → track → hành vi",col:1,row:2},
  {id:"outcomes",title:"Đầu ra nghiên cứu",sub:"Tần suất · thời lượng · minh chứng",col:3,row:2},
  {id:"quality",title:"Kiểm chứng & đánh giá",sub:"Ground truth · độ trễ · ổn định",col:1,row:3,input:"Dữ liệu test độc lập, nhãn chuẩn được kiểm duyệt và cấu hình thực nghiệm.",process:"Đánh giá detection, tracking và hành vi riêng; đo cold start, độ trễ sau warm-up, RAM/VRAM trên máy thật.",output:"Chất lượng và giới hạn có căn cứ; mức phù hợp với đánh giá của giảng viên.",note:"Chưa dùng FPS từ máy khác, box nội suy hoặc smoke test ngắn để khẳng định Recall hay chất lượng dài hạn.",status:"Cần thực nghiệm"},
  {id:"privacy",title:"Minh bạch & riêng tư",sub:"Quyền sử dụng · quyền truy cập",col:3,row:3,input:"Dữ liệu lớp học có quyền sử dụng, phạm vi và mục đích rõ ràng.",process:"Giới hạn truy cập, lưu dữ liệu phù hợp; tách nhãn nháp và GT, quan sát và suy luận. Rà soát trước khi dùng dịch vụ ngoài.",output:"Quy trình có thể đối chiếu và dữ liệu được quản lý theo mục đích.",note:"ID tracking cục bộ không đồng nghĩa dữ liệu đã ẩn danh; video vẫn có thể chứa người nhận diện được.",status:"Nguyên tắc xuyên suốt"}
];
Object.assign(mindBranches[1],{...nodes[0],id:"data",title:"Dữ liệu đầu vào",short:"",col:3,row:1});
Object.assign(mindBranches[2],{...nodes[2],id:"models",title:"Mô hình & pipeline",col:1,row:2,note:"Phát hiện người và ByteTrack/Full đã có; nhận dạng hành vi theo thời gian đang nghiên cứu. Không mặc định một công cụ dự kiến là lựa chọn đã chốt."});
Object.assign(mindBranches[3],{...nodes[6],id:"outcomes",title:"Đầu ra nghiên cứu",col:3,row:2});

const wireSVG = `<svg class="flow-wires" viewBox="0 0 1000 600" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="arrow-ready" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#7c9c7c" stroke-width="1.2"/></marker><marker id="arrow-next" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#c09967" stroke-width="1.2"/></marker></defs><g fill="none" stroke="#8ca58a" stroke-width="2" marker-end="url(#arrow-ready)"><path d="M278 93H361"/><path d="M631 93H714"/><path d="M853 179V210"/></g><g fill="none" stroke="#c09d70" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow-next)"><path d="M714 303H639"/><path d="M361 303H286"/><path d="M147 388V513H361"/></g></svg>`;

function renderDiagrams() {
  $("#flow-diagram").innerHTML = wireSVG + nodes.map((node,index)=>`<button class="flow-node ${node.planned?"planned":""} ${index===6?"last-node":""}" style="left:${node.x}%;top:${node.y}%" data-node="${node.id}" aria-pressed="false"><span class="node-topline">${String(index+1).padStart(2,"0")}<span class="node-icon" aria-hidden="true">${node.icon}</span></span><span class="node-title">${node.title}</span><span class="node-subtitle">${node.short}</span></button>`).join("");
  $("#mind-diagram").innerHTML = `<svg class="mind-lines" viewBox="0 0 600 360" preserveAspectRatio="none" aria-hidden="true"><g fill="none" stroke="#aac0a1" stroke-width="1.4"><path d="M300 180C205 180 220 60 100 60"/><path d="M300 180C395 180 380 60 500 60"/><path d="M300 180H100"/><path d="M300 180H500"/><path d="M300 180C205 180 220 300 100 300"/><path d="M300 180C395 180 380 300 500 300"/></g></svg><div class="mind-center">CLASSROOM AI<br>Video → tương tác lớp</div>` + mindBranches.map(branch=>`<button class="mind-branch" data-node="${branch.id}" style="grid-column:${branch.col};grid-row:${branch.row}" aria-pressed="false"><strong>${escapeHTML(branch.title)}</strong><small>${escapeHTML(branch.sub || branch.short || "Chọn để khám phá")}</small></button>`).join("");
  $$('[data-node]').forEach(button=>button.addEventListener("click",()=>selectNode(button.dataset.node)));
  selectNode("video");
}

function selectNode(id) {
  const node = [...nodes,...mindBranches].find(item=>item.id===id);
  if (!node) return;
  $$('[data-node]').forEach(button=>{const selected=button.dataset.node===id;button.classList.toggle("selected",selected);button.setAttribute("aria-pressed",String(selected));});
  const index=nodes.indexOf(node);
  $("#node-detail").innerHTML=`<span class="detail-number">${index>=0?`KHỐI ${String(index+1).padStart(2,"0")} / 07`:"NHÁNH NGHIÊN CỨU"}</span><h3>${escapeHTML(node.title)}</h3><span class="pill detail-status ${node.planned?"pending":""}">${escapeHTML(node.status)}</span>${[["ĐẦU VÀO",node.input],["XỬ LÝ / NỘI DUNG",node.process],["ĐẦU RA",node.output]].map(([label,value])=>`<div class="detail-block"><strong>${label}</strong><p>${escapeHTML(value)}</p></div>`).join("")}<p class="detail-note">↳ ${escapeHTML(node.note)}</p>`;
}

function activateView(view) {
  $$('[data-view]').forEach(button=>{const selected=button.dataset.view===view;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;});
  ["flow","mind","poster"].forEach(id=>{$(`#view-${id}`).hidden=id!==view;});
  $("#node-detail").hidden=view==="poster";
  $(".map-workspace").style.gridTemplateColumns=view==="poster"?"minmax(0,1fr)":"";
  if(view==="flow")selectNode("video");
  if(view==="mind")selectNode("scope");
}

$$('[data-view]').forEach((button,index)=>{
  button.addEventListener("click",()=>activateView(button.dataset.view));
  button.addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
    event.preventDefault();
    const tabs=$$('[data-view]');
    const next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;
    activateView(tabs[next].dataset.view);tabs[next].focus();
  });
});

let toastTimer;
function showToast(message) {const toast=$("#toast");toast.textContent=message;toast.classList.add("visible");clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.classList.remove("visible"),3500);}
function downloadBlob(blob,name) {const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);}

function exportSVG(view="flow") {
  const heading = `<text x="70" y="75" fill="#18333c" font-size="32" font-weight="700">CLASSROOM AI · ${view==="mind"?"SƠ ĐỒ TƯ DUY":"LUỒNG XỬ LÝ DỰ ÁN"}</text><text x="70" y="111" fill="#687c6c" font-size="15">Video lớp học → hành vi quan sát được → tổng hợp tương tác của lớp theo thời gian</text>`;
  const footer = `<text x="70" y="790" fill="#687c6c" font-size="12">Nét liền: đã có trong pipeline · Nét đứt: hướng nghiên cứu · Tham chiếu tasks/plan.md và tasks/project_memory.md</text>`;
  let diagram="";
  if(view==="mind") {
    const centerX=650,centerY=450;
    const positions=[[190,230],[1110,230],[190,450],[1110,450],[190,670],[1110,670]];
    diagram=mindBranches.map((branch,i)=>{const [x,y]=positions[i];return `<path d="M${centerX} ${centerY}C${(x+centerX)/2} ${centerY} ${(x+centerX)/2} ${y} ${x} ${y}" fill="none" stroke="#abc1a1" stroke-width="2"/><rect x="${x-140}" y="${y-50}" width="280" height="100" rx="9" fill="#eff4e9" stroke="#a9c09f"/><text x="${x}" y="${y-7}" fill="#244634" font-size="19" text-anchor="middle" font-weight="600">${escapeHTML(branch.title)}</text><text x="${x}" y="${y+22}" fill="#788c70" font-size="12" text-anchor="middle">${escapeHTML(branch.sub || branch.short || "Chi tiết trên web local")}</text>`;}).join("") + `<circle cx="${centerX}" cy="${centerY}" r="112" fill="#18333c" stroke="#ccdac3" stroke-width="12"/><text x="${centerX}" y="${centerY-8}" fill="#eef5e6" font-size="26" font-weight="700" text-anchor="middle">CLASSROOM AI</text><text x="${centerX}" y="${centerY+28}" fill="#afc8ad" font-size="15" text-anchor="middle">Tương tác của lớp</text>`;
  } else {
    const connectors=`<defs><marker id="end" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="none" stroke="#789c78" stroke-width="1.5"/></marker></defs><g transform="translate(70 150) scale(1.15 .98)"><g fill="none" stroke="#8ca58a" stroke-width="2" marker-end="url(#end)"><path d="M278 93H361"/><path d="M631 93H714"/><path d="M853 179V210"/></g><g fill="none" stroke="#c09d70" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#end)"><path d="M714 303H639"/><path d="M361 303H286"/><path d="M147 388V513H361"/></g></g>`;
    diagram=connectors+nodes.map((node,index)=>{const x=70+node.x*11.5,y=150+node.y*5.88;return `<g><rect x="${x}" y="${y}" width="292" height="158" rx="8" fill="${node.planned?"#fbf4e7":"#edf4e8"}" stroke="${node.planned?"#c7a374":"#9cb896"}" stroke-width="1.5" ${node.planned?'stroke-dasharray="7 5"':""}/><text x="${x+19}" y="${y+28}" fill="#81947a" font-size="12">${String(index+1).padStart(2,"0")} / ${escapeHTML(node.status)}</text><text x="${x+19}" y="${y+75}" fill="#234736" font-size="20" font-weight="600">${escapeHTML(node.title)}</text><text x="${x+19}" y="${y+109}" fill="#81947a" font-size="13">${escapeHTML(node.short)}</text></g>`;}).join("");
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="830" viewBox="0 0 1300 830" font-family="Segoe UI,Arial,sans-serif"><rect width="1300" height="830" fill="#f9fbf5"/>${heading}${diagram}${footer}</svg>`;
}

$("#download-svg").addEventListener("click",()=>{const view=$('[data-view][aria-selected="true"]').dataset.view;if(view==="poster"){const a=document.createElement("a");a.href="assets/project-overview.png";a.download="so-do-tong-quan-du-an.png";a.click();return;}downloadBlob(new Blob([exportSVG(view)],{type:"image/svg+xml;charset=utf-8"}),`classroom-ai-${view}.svg`);showToast("Đã xuất sơ đồ SVG có thể chỉnh sửa.");});
$("#download-png").addEventListener("click",async()=>{
  const view=$('[data-view][aria-selected="true"]').dataset.view;
  if(view==="poster"){const a=document.createElement("a");a.href="assets/project-overview.png";a.download="so-do-tong-quan-du-an.png";a.click();return;}
  const image=new Image();const url=URL.createObjectURL(new Blob([exportSVG(view)],{type:"image/svg+xml;charset=utf-8"}));
  image.onload=()=>{const canvas=document.createElement("canvas");canvas.width=2600;canvas.height=1660;const context=canvas.getContext("2d");context.scale(2,2);context.drawImage(image,0,0);URL.revokeObjectURL(url);canvas.toBlob(blob=>{if(blob){downloadBlob(blob,`classroom-ai-${view}.png`);showToast("Đã xuất sơ đồ PNG 2600 × 1660.");}else showToast("Không xuất được PNG; hãy tải SVG.");},"image/png");};
  image.onerror=()=>{URL.revokeObjectURL(url);showToast("Không xuất được PNG; hãy tải SVG.");};image.src=url;
});

$$('[data-copy]').forEach(button=>button.addEventListener("click",async()=>{
  try{if(!navigator.clipboard?.writeText)throw new Error("clipboard unavailable");await navigator.clipboard.writeText(button.dataset.copy);showToast("Đã sao chép lệnh.");}
  catch{const area=document.createElement("textarea");area.value=button.dataset.copy;document.body.append(area);area.select();const copied=document.execCommand("copy");area.remove();showToast(copied?"Đã sao chép lệnh.":"Hãy chọn và sao chép lệnh trong khung.");}
}));
$("#print-button").addEventListener("click",()=>window.print());
$("#sample-path").addEventListener("click",()=>{$("#video-path").value="C:\\Annotation_Phase1\\video\\VCP2_204_20260810_V01.mp4";$("#video-path").focus();showToast("Đã điền đường dẫn trong ảnh; cần có file thật trên máy chủ.");});

const number=(value,digits=2)=>Number.isFinite(Number(value))&&value!==null&&value!==""?Number(value).toLocaleString("vi-VN",{maximumFractionDigits:digits}):"Không có thông tin";
function duration(value){if(value===null||value===undefined||!Number.isFinite(Number(value)))return "Không có thông tin";const seconds=Math.round(Number(value));return `${Math.floor(seconds/3600).toString().padStart(2,"0")}:${Math.floor(seconds%3600/60).toString().padStart(2,"0")}:${(seconds%60).toString().padStart(2,"0")}`;}
function renderProbe(data) {
  const formats=String(data.container_formats||data.container||"?");
  const containerLabel=formats.includes("mov")&&formats.includes("mp4")?`MOV / MP4${data.major_brand?` · ${data.major_brand.trim()}`:""}`:`${data.container ?? "?"}${data.container==="mpeg"?" (MPEG-PS)":""}`;
  const rows=[["Container thực",containerLabel],["Codec / profile",`${(data.codec||"?").toUpperCase()}${data.profile?` · ${data.profile}`:""}${data.level_label?` · ${data.level_label}`:""}`],["Độ phân giải",data.width&&data.height?`${data.width} × ${data.height}`:"Không có thông tin"],["Tốc độ khung hình",data.fps?`${number(data.fps,4)} FPS`:"Không có thông tin"],["Thời lượng",duration(data.duration_seconds)],["Dung lượng",data.size_bytes?`${number(data.size_bytes/1024/1024)} MiB`:"Không có thông tin"],["Bitrate tổng",data.bit_rate?`${number(data.bit_rate/1000)} kb/s`:"Không có thông tin"],["Pixel / màu",[data.pixel_format,data.color_range,data.color_space].filter(Boolean).join(" · ")||"Không có thông tin"]];
  const streams=(data.streams||[]).map(stream=>`${stream.type||"?"}: ${stream.codec||"?"}`).join(" · ");
  $("#probe-result").innerHTML=`<p class="probe-success">✓ Đã đọc file thật: ${escapeHTML(data.name)}</p><div class="probe-meta">${rows.map(([label,value])=>`<div><span>${label}</span><strong>${escapeHTML(value)}</strong></div>`).join("")}</div><p class="small muted" style="margin-top:12px">Luồng: ${escapeHTML(streams||"Không có thông tin")}</p>${(data.warnings||[]).map(warning=>`<p class="probe-warnings">↳ ${escapeHTML(warning)}</p>`).join("")}<p class="small muted">Metadata không xác nhận CFR, khả năng giải mã toàn bộ hoặc chất lượng nhận dạng. Chạy pipeline video để kiểm nguồn chi tiết.</p><details><summary class="small">Xem metadata JSON</summary><pre class="raw-metadata">${escapeHTML(JSON.stringify(data,null,2))}</pre></details>`;
}

$("#probe-form").addEventListener("submit",async event=>{
  event.preventDefault();const button=$("#probe-button");button.disabled=true;button.textContent="Đang đọc metadata…";$("#probe-result").textContent="Đang mở ffprobe, tối đa 30 giây…";
  try{const response=await fetch("/api/probe",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:$("#video-path").value.trim()}),signal:AbortSignal.timeout(35000)});const data=await response.json();if(!response.ok||!data.ok)throw new Error(data.error||"Không đọc được video.");renderProbe(data);}
  catch(error){$("#probe-result").innerHTML=`<p class="probe-error">${escapeHTML(error.name==="TimeoutError"?"Hết thời gian chờ. Hãy kiểm tra file hoặc thử video ngắn hơn.":error.message)}</p><p class="small muted">Đường dẫn phải thuộc máy đang chạy web. Xem công cụ qua http://127.0.0.1:8765 để đọc file local.</p>`;}
  finally{button.disabled=false;button.innerHTML='Đọc thông số thật <span>→</span>';}
});

let sourceDocuments=[];
async function loadProject() {
  try{const response=await fetch("/api/project");if(!response.ok)throw new Error("Không đọc được tài liệu chính thức.");const data=await response.json();$("#official-title").textContent=data.title;$("#official-objective").textContent=data.objective;$("#official-research").innerHTML=(data.research_items||[]).map(item=>`<li>${escapeHTML(item)}</li>`).join("");sourceDocuments=data.sources||[];$("#source-list").innerHTML=sourceDocuments.map((source,index)=>`<button class="source-item" data-source="${index}"><span class="doc-icon" aria-hidden="true">▤</span><span><strong>${escapeHTML(source.title)}</strong><small>${escapeHTML(source.path)}</small></span><span class="doc-arrow" aria-hidden="true">↗</span></button>`).join("");$$('[data-source]').forEach(button=>button.addEventListener("click",()=>{const source=sourceDocuments[Number(button.dataset.source)];$("#source-dialog-title").textContent=source.title;$("#source-dialog-path").textContent=source.path;$("#source-dialog-content").textContent=source.content;$("#source-dialog").showModal();}));}
  catch(error){$("#official-title").textContent="Mở web bằng run_project_site.cmd để đọc tên đề tài từ tasks/plan.md.";$("#official-research").innerHTML="<li>Chưa kết nối được tài liệu gốc. Hãy mở bằng máy chủ local.</li>";$("#source-list").textContent=error.message;}
}
async function checkHealth(){try{const response=await fetch("/api/health");if(!response.ok)throw new Error();const health=await response.json();$("#connection").textContent=health.probe_available?"Local · ffprobe sẵn sàng":"Local · cần ffprobe để đọc video";}catch{$("#connection").textContent="Chưa kết nối máy chủ";$("#connection").style.color="#ad7055";}}
$("#close-dialog").addEventListener("click",()=>$("#source-dialog").close());
$("#source-dialog").addEventListener("click",event=>{const box=event.currentTarget.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)event.currentTarget.close();});

const navLinks=$$('.main-nav a');
if("IntersectionObserver" in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){navLinks.forEach(link=>link.classList.toggle("active",link.hash===`#${entry.target.id}`));}}},{rootMargin:"-10% 0px -65% 0px",threshold:0});$$('main>section').forEach(section=>observer.observe(section));}
renderDiagrams();
activateView("mind");
Promise.allSettled([loadProject(),checkHealth()]);

// Reading paths connect the current code to the proposed research work.
const futureStages = {
  data: {
    title:"Làm rõ dữ liệu và nhãn trước",status:"Đã thử công cụ · quy trình GT còn cần hoàn thiện",
    before:"Có video được phép sử dụng, đọc được thông số và các đoạn thể hiện hành vi cần quan sát. Ảnh MediaInfo hiện là nguồn tham chiếu cho định dạng; cần file thật để kiểm tra.",
    work:"Viết quy tắc cho từng hành vi, thời điểm bắt đầu/kết thúc, trường hợp không chắc và cách gán nhãn khi bị che. Tách bbox người, ID cục bộ và khoảng hành vi. Chia dữ liệu theo nguồn hoặc buổi học để hạn chế rò rỉ frame gần nhau.",
    output:"Video/đoạn mẫu có mã tham chiếu; bảng nhãn, quy tắc gán nhãn, danh sách train/validation/test và nhãn chuẩn đã được người kiểm duyệt.",
    evidence:"Kiểm tra độc lập một phần nhãn và ghi các trường hợp bất đồng. Prediction AI chỉ trở thành ground truth sau khi được sửa và duyệt; số task đã import chưa chứng minh chất lượng bộ dữ liệu.",
    open:"Chốt công cụ và quy tắc review, quy mô pilot, cách chia nguồn, cách xử lý hành vi mơ hồ. Label Studio đã được thử riêng; chưa coi đó là lựa chọn cuối cho toàn bộ nghiên cứu.",
    code:"video/media.py đọc nguồn; datasets.py tổ chức dữ liệu detector; việc gán nhãn khoảng hành vi hiện chưa có pipeline hoàn chỉnh trong ứng dụng."
  },
  behavior: {
    title:"Nhận dạng hành vi từ một đoạn thời gian",status:"Chưa có bộ nhận dạng hành vi hoàn chỉnh",
    before:"Có định nghĩa và nhãn chuẩn cho giơ tay, đứng lên, di chuyển lên bảng và viết bảng; chọn được các đoạn test độc lập.",
    work:"Làm baseline nhỏ, xem có cần tư thế, hình ảnh hay kết hợp cả hai. Khảo sát YOLO Pose, RTMPose, ST-GCN, PoseC3D, Jev theo dữ liệu thực tế; so sánh lỗi trước khi chọn kiến trúc.",
    output:"Sự kiện gồm loại hành vi, mốc bắt đầu/kết thúc, độ tin cậy hoặc trạng thái chưa chắc, và vị trí video để xem lại minh chứng.",
    evidence:"Đối chiếu với sự kiện do người duyệt: đo bỏ sót, báo nhầm, đếm lặp và sai ranh giới thời gian. Lưu cả ví dụ nhận sai, không chỉ đoạn nhận đúng.",
    open:"Chưa chốt mô hình, đầu vào tư thế/hình ảnh, độ dài đoạn quan sát hoặc ngưỡng chấp nhận. Mục tiêu trễ dưới 3 giây không buộc lịch sử quan sát chỉ dài 3 giây.",
    code:"inference/detector.py và video/pipeline.py hiện cung cấp detection; video/tracking.py liên kết người trong Full. Chưa có module classifier hành vi được xác nhận trong phạm vi này."
  },
  aggregation: {
    title:"Từ sự kiện sang tổng hợp của lớp",status:"Đầu ra nghiên cứu · chưa triển khai",
    before:"Có sự kiện hành vi đã được kiểm tra, biết thời gian quan sát hợp lệ và các đoạn mất/không đủ dữ liệu.",
    work:"Hạn chế đếm lặp cùng một sự kiện, tính tần suất và thời lượng theo từng khoảng, rồi tổng hợp cả buổi. Giữ liên kết tới minh chứng; tách không quan sát được khỏi không có hành vi.",
    output:"Bảng hoặc biểu đồ diễn biến của cả lớp theo thời gian; tần suất, thời lượng, ngữ cảnh và các mốc video. Phân mức tương tác chỉ là nhận định tham khảo khi tiêu chí đã được kiểm định.",
    evidence:"So sánh số sự kiện/thời lượng với tổng hợp thủ công trên cùng đoạn; đối chiếu cách diễn giải với giảng viên và ghi các trường hợp chưa phù hợp.",
    open:"Chốt độ dài khoảng tổng hợp, cách chuẩn hóa khi số người nhìn thấy thay đổi, bộ tiêu chí và ngưỡng phân mức. Không yêu cầu hồ sơ từng sinh viên hoặc ID xuyên suốt buổi.",
    code:"video/tracked_processing.py và tracking.py có thể cung cấp track để hạn chế đếm lặp. report.json hiện là báo cáo kỹ thuật của lượt chạy; chưa phải báo cáo tương tác lớp."
  },
  validation: {
    title:"Kiểm chứng bằng video lớp học thực tế",status:"Cần thực nghiệm theo từng đầu ra",
    before:"Có tập test độc lập, nhãn chuẩn đã duyệt, cấu hình và phiên bản trọng số rõ ràng.",
    work:"Đánh giá detection, tracking và hành vi riêng. Đo khởi động, độ trễ khi model sẵn sàng, RAM/VRAM và độ ổn định trên đúng máy/video dự kiến dùng. Đối chiếu phần tổng hợp với giảng viên.",
    output:"Báo cáo có cấu hình, dữ liệu test, cách đo, kết quả, ví dụ sai và giới hạn. Giữ liên kết từ con số về lượt chạy hoặc đoạn video tương ứng.",
    evidence:"Precision/Recall cần nhãn detection; IDF1 cần nhãn tracking; hành vi cần nhãn sự kiện. Kiểm thử pipeline hoặc frame tổng hợp không thay thế các phép đo này.",
    open:"Chốt tập test và tiêu chí nghiệm thu. Các số từ CPU/T4 hoặc clip ngắn không tự trở thành kết quả trên RTX 5060 Laptop, video dài hay lớp học khác.",
    code:"video/metrics.py và đầu ra report.json phục vụ số đo kỹ thuật; evaluation.py và experiments/ chứa các công cụ đánh giá detector/thí nghiệm theo phạm vi từng công cụ."
  },
  integration: {
    title:"Đưa kết quả tới giảng viên và hệ sinh thái AI-Native",status:"Mục tiêu tích hợp · chưa chốt cách triển khai",
    before:"Đầu ra sự kiện/tổng hợp có cấu trúc ổn định, có bằng chứng chất lượng và quy định truy cập dữ liệu.",
    work:"Nghiên cứu giao diện xem diễn biến theo khoảng và cả buổi, bấm về minh chứng, xem giới hạn của nhận định. Xác định điểm tích hợp AI-Native và cách trao đổi dữ liệu trước khi nối dịch vụ.",
    output:"Giao diện và hợp đồng dữ liệu có phiên bản, kết quả có thể xem lại, quyền truy cập phù hợp và cơ chế để giảng viên phản hồi.",
    evidence:"Thử tác vụ thực tế: tìm một khoảng ít/nhiều sự kiện, xem lại video minh chứng, hiểu căn cứ của phần tổng hợp và phản hồi khi nhận định chưa phù hợp.",
    open:"Chưa chốt nền tảng AI-Native, cách tích hợp hoặc cơ chế phân quyền triển khai. Web thuyết minh này đang giải thích dự án và đọc metadata, chưa thực hiện các chức năng báo cáo hành vi đó.",
    code:"interfaces/web.py và web.html là web video local dùng chung pipeline; interfaces/streamlit_ui/ giữ công cụ nghiên cứu trước đây. docs/project_site/ là web thuyết minh riêng. Việc đánh giá mức tương tác và tích hợp dịch vụ vẫn cần kiểm chứng và thiết kế thêm."
  }
};

function selectFutureStage(id) {
  const stage=futureStages[id]; if(!stage||!$("#future-detail"))return;
  $$('[data-future-stage]').forEach(button=>{const active=button.dataset.futureStage===id;button.classList.toggle("selected",active);button.setAttribute("aria-pressed",String(active));});
  $("#future-detail").innerHTML=`<div class="future-detail-heading"><span class="pill pending">${escapeHTML(stage.status)}</span><h3>${escapeHTML(stage.title)}</h3></div><div class="future-content">${[["Cần có trước",stage.before],["Mình dự kiến làm gì",stage.work],["Đầu ra cần nhìn thấy",stage.output],["Căn cứ để đánh giá",stage.evidence],["Những việc chưa chốt",stage.open],["Code hiện có liên quan",stage.code]].map(([title,text])=>`<article><h4>${title}</h4><p>${escapeHTML(text)}</p></article>`).join("")}</div>`;
}
$$('[data-future-stage]').forEach(button=>button.addEventListener("click",()=>selectFutureStage(button.dataset.futureStage)));
selectFutureStage("data");

let codeModules=[];
let activeCodeGroup="all";
let activeCodeId=null;
const codeGroupLabels={video:"Video",interface:"Giao diện",inference:"Suy luận",training:"Dữ liệu & huấn luyện",site:"Web thuyết minh"};
const normalizeSearch=value=>String(value||"").normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/đ/g,"d").toLocaleLowerCase("vi");

function renderCodeList() {
  if(!$("#code-list"))return;
  const query=normalizeSearch($("#code-search")?.value);
  const visible=codeModules.filter(module=>(activeCodeGroup==="all"||module.group===activeCodeGroup)&&normalizeSearch([module.path,module.title,module.purpose,...(module.symbols||[])].join(" ")).includes(query));
  $("#code-list").innerHTML=visible.length?visible.map(module=>`<button class="code-module ${module.id===activeCodeId?"selected":""}" data-code-id="${escapeHTML(module.id)}" aria-pressed="${module.id===activeCodeId}"><span class="code-module-group">${escapeHTML(codeGroupLabels[module.group]||module.group)}</span><strong>${escapeHTML(module.title)}</strong><code>${escapeHTML(module.path)}</code></button>`).join(""):"<p class=\"muted small\">Không thấy file khớp bộ lọc. Thử từ như video, tracking, model, giao diện hoặc server.</p>";
  $$('[data-code-id]').forEach(button=>button.addEventListener("click",()=>selectCodeModule(button.dataset.codeId)));
  if(visible.length&&!visible.some(module=>module.id===activeCodeId))selectCodeModule(visible[0].id);
  if(!visible.length&&$("#code-detail"))$("#code-detail").innerHTML="<p class=\"muted\">Chọn lại nhóm hoặc từ khóa để xem một file.</p>";
  if($("#code-count"))$("#code-count").textContent=`${visible.length} / ${codeModules.length} file chính`;
}

function selectCodeModule(id) {
  const module=codeModules.find(item=>item.id===id);if(!module||!$("#code-detail"))return;
  activeCodeId=id;
  $$('[data-code-id]').forEach(button=>{const active=button.dataset.codeId===id;button.classList.toggle("selected",active);button.setAttribute("aria-pressed",String(active));});
  $("#code-detail").innerHTML=`<div class="code-detail-top"><span class="pill">FILE HIỆN CÓ${module.file_exists===false?" · KHÔNG TÌM THẤY":""}</span><h3>${escapeHTML(module.title)}</h3><code>${escapeHTML(module.path)}</code></div><p class="code-purpose">${escapeHTML(module.purpose)}</p><div class="code-io"><article><h4>Nhận gì?</h4><p>${escapeHTML(Array.isArray(module.inputs)?module.inputs.join("; "):module.inputs)}</p></article><article><h4>Tạo ra / chuyển tiếp gì?</h4><p>${escapeHTML(Array.isArray(module.outputs)?module.outputs.join("; "):module.outputs)}</p></article></div><div class="code-symbols"><h4>Chỗ nên đọc trong file</h4>${(module.symbols||[]).map(symbol=>`<code>${escapeHTML(symbol)}</code>`).join(" ")}</div><button id="view-code-source" class="small-button" ${module.file_exists===false?"disabled":""}>Đọc mã nguồn hiện tại ↗</button><p class="small muted code-source-note">Đường dẫn tính từ thư mục gốc BTL-AI-Camera-cd8. Danh sách này mô tả các file đang có; các module hành vi/tổng hợp tương lai chưa được coi là đã triển khai.</p>`;
  $("#view-code-source").addEventListener("click",()=>openCodeSource(module));
}

async function openCodeSource(module) {
  const button=$("#view-code-source");button.disabled=true;button.textContent="Đang đọc file…";
  try{const response=await fetch(`/api/code/${encodeURIComponent(module.id)}`);const data=await response.json();if(!response.ok)throw new Error(data.error||"Không đọc được file.");$("#source-dialog-title").textContent=module.title;$("#source-dialog-path").textContent=`${data.path} · ${data.line_count??"?"} dòng${data.truncated?" · trích đoạn":""}`;$("#source-dialog-content").textContent=String(data.content||"").split("\n").map((line,index)=>`${String(index+1).padStart(4," ")}  ${line}`).join("\n");$("#source-dialog").showModal();}
  catch(error){showToast(error.message);}
  finally{button.disabled=false;button.textContent="Đọc mã nguồn hiện tại ↗";}
}

async function loadCodeMap() {
  if(!$("#code-list"))return;
  try{const response=await fetch("/api/code");if(!response.ok)throw new Error("Chưa đọc được bản đồ code; cần khởi động lại máy chủ sau khi cập nhật.");const data=await response.json();codeModules=data.modules||[];renderCodeList();}
  catch(error){$("#code-list").textContent=error.message;}
}
$("#code-search")?.addEventListener("input",renderCodeList);
$$('[data-code-group]').forEach(button=>button.addEventListener("click",()=>{activeCodeGroup=button.dataset.codeGroup;$$('[data-code-group]').forEach(item=>item.setAttribute("aria-pressed",String(item===button)));renderCodeList();}));
loadCodeMap();
