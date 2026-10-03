/* Plain-text, local-only views of recorded experiment evidence. */
'use strict';
function format(value, digits=3) {
  if (value === null || value === undefined || (typeof value === 'number' && !Number.isFinite(value))) return '未记录';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  if (typeof value === 'boolean') return value ? '是' : '否';
  return typeof value === 'string' ? value : JSON.stringify(value);
}
function setText(element, value) { if (element) element.textContent = format(value); }
function mergeEvents(previous, incoming, reset=false) {
  const map = new Map((reset ? [] : previous).map(r => [r.cursor, r]));
  for (const row of incoming) map.set(row.cursor, row);
  return [...map.values()].sort((a,b)=>a.cursor-b.cursor).slice(-5000);
}
class RequestSequence {
  constructor(){this.version=0;}
  next(){return ++this.version;}
  isCurrent(ticket){return ticket===this.version;}
}
function currentTaskState(latest,task) {
  for(const kind of ['agent_state','task_result']) {
    const payload=latest[kind];if(!payload)continue;
    const state=payload.state||payload;
    if(payload.task_id===task.task_id || (!payload.task_id &&
       !['incomplete','running'].includes(task.status) && state.status===task.status))return state;
  }
  return {};
}
function candidateViews(plan) {
  const evidence=plan.evidence||plan.planning_evidence||{};
  return (evidence.candidates||[]).map(c=>({id:c.candidate??c.candidate_id,
    skill:c.skill??null,score:c.score??c.cost??null,prefix:c.trusted_prefix??null,
    chosen:(c.candidate??c.candidate_id)===evidence.selected_candidate,rejection:c.rejection??null}));
}
function taskGraphRows(state) {
  return Object.entries(state.nodes||{}).map(([id,n])=>[id,n.status??null,n.actions??null,n.hold_count??null,n.recoveries??null]);
}
function clearEvidenceViews(getElement) {
  const messages={eventDetail:'未选择事件',eventDetailTitle:'事件详情',artifactDetail:'未选择工件',
    artifactTitle:'配置、来源与结果',predictionAlignment:'选择工件查看图像证据',
    predictionStepLabel:'未记录',predictionCaption:'模型预测',frameAlignment:'未选择事件，无法对齐',
    frameMeta:'',stateRows:'',uncertaintyNote:'没有对应的预测记录'};
  for (const [id,text] of Object.entries(messages)) setText(getElement(id),text);
  for (const id of ['showEventEvidence','beforeImage','predictedImage','observedImage','trainingChart']) {
    const el=getElement(id);if(el)el.hidden=true;
  }
}
function alignment(frame, event) {
  if (!frame?.episode_id || !event?.episode_id) return 'unknown';
  if (frame.episode_id !== event.episode_id) return 'different_episode';
  const step = event.step_id ?? event.observation_step;
  if (!Number.isInteger(step) || !Number.isInteger(frame.step_id)) return 'unknown';
  if (step !== frame.step_id) return 'different_step';
  if (!Number.isFinite(frame.sim_time_s) || !Number.isFinite(event.sim_time_s)) return 'same_step';
  return Math.abs(frame.sim_time_s-event.sim_time_s) < 1e-8 ? 'exact' : 'different_time';
}
if (typeof module !== 'undefined') module.exports = {format, setText, mergeEvents, alignment, candidateViews, taskGraphRows, clearEvidenceViews, RequestSequence, currentTaskState};
if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const names = {task_started:'任务开始',task_result:'任务结果',plan:'计划与模型评估',feedback:'真实反馈',
    prediction_residual:'预测残差',execution_started:'执行开始',execution_failed:'执行失败',execution:'执行结果',
    command_sent:'命令发送',command_ack:'执行确认',command_uncertain:'执行结果不确定',replan_requested:'请求重规划',
    observation:'环境观测',agent_state:'Agent 状态',language_planning:'语言规划',mission_validated:'任务校验',
    mission_goal:'子目标',mission_result:'任务结果',goal_result:'目标结果',executed_transition:'真实状态转移',
    safety_stop:'安全停止',failure:'异常',session_exit:'会话退出',skill_result:'技能结果',
    node_state:'任务节点',predicate_result:'真实条件验证',recovery_started:'开始恢复',
    recovery_result:'恢复结果',execution_stopped:'动作段提前停止'};
  const statusNames = {running:'运行中',succeeded:'成功',failed:'失败',incomplete:'未完成 / 缺少终态',
    budget_exhausted:'预算耗尽',needs_review:'需要检查',execution_failed:'执行失败',safety_stop:'安全停止',
    fault_latched:'故障锁止',recovering:'有限恢复中',canceled:'已取消'};
  let runs=[], selected=null, records=[], stream=null, cursor=0, generation=0, summary={}, artifacts=[], prediction=null;
  let currentFrame=null, frameURL=null, frameBusy=false, selectedEvent=null, selectedArtifact='', artifactVersion=0;
  const detailRequests=new RequestSequence();
  const element = (tag, text, cls) => { const node=document.createElement(tag); if(text!==undefined)setText(node,text); if(cls)node.className=cls; return node; };
  const pretty = value => JSON.stringify(value ?? '未记录', null, 2);
  const notice = message => { $('notice').hidden=!message; setText($('notice'),message); };
  async function getJSON(url) { const response=await fetch(url,{cache:'no-store'}); if(!response.ok)throw new Error(`读取失败 (${response.status})`); return response.json(); }
  function badge(id, text, kind='') { setText($(id), text); $(id).className='badge '+kind; }
  function pairs(node, values, compact=false) {
    node.replaceChildren();
    for(const [key,value] of values){
      const dt=element('dt',key),dd=element('dd',format(value));
      if(compact){const wrap=element('div');wrap.append(dt,dd);node.append(wrap);}else node.append(dt,dd);
    }
  }
  function cells(parent, values, cls='') { const row=element('tr',undefined,cls); values.forEach(value=>row.append(element('td',format(value))));parent.append(row);return row; }
  function page(name){document.querySelectorAll('[data-view]').forEach(n=>n.hidden=n.dataset.view!==name);document.querySelectorAll('[data-page]').forEach(n=>n.classList.toggle('active',n.dataset.page===name));}
  document.querySelectorAll('[data-page]').forEach(n=>n.addEventListener('click',()=>page(n.dataset.page)));
  async function refreshRuns() {
    try {
      const data=await getJSON('/api/runs');runs=data.runs;renderRuns();
      for(const id of ['compareA','compareB']) {
        const value=$(id).value;$(id).replaceChildren(...runs.map(r=>{const option=element('option',r.run_path);option.value=r.run_id;return option;}));
        if(runs.some(r=>r.run_id===value))$(id).value=value;
      }
      if(!selected&&runs.length){const hinted=new URLSearchParams(location.search).get('run');chooseRun(runs.some(r=>r.run_id===hinted)?hinted:runs[runs.length-1].run_id);}
      if(!runs.length)setText($('runList'),'尚无实验记录。启动实验后刷新即可。');
    }catch(error){badge('connection','服务连接失败','bad');notice(error.message);}
  }
  function renderRuns() {
    const search=$('runSearch').value.toLowerCase();$('runList').replaceChildren();
    runs.filter(r=>r.run_path.toLowerCase().includes(search)).forEach(run=>{
      const button=element('button',undefined,'run-button'+(selected===run.run_id?' selected':''));
      button.append(element('span',run.run_path),element('small',run.run_kind||'实验 / 历史工件'));button.onclick=()=>chooseRun(run.run_id);$('runList').append(button);
    });
  }
  async function chooseRun(runId) {
    selected=runId;generation++;const token=generation;stream?.close();records=[];cursor=0;summary={};prediction=null;selectedEvent=null;selectedArtifact='';artifactVersion++;
    clearEvidenceViews($);
    detailRequests.next();
    currentFrame=null;if(frameURL){URL.revokeObjectURL(frameURL);frameURL=null;}$('liveFrame').hidden=true;$('framePlaceholder').hidden=false;
    for(const id of ['beforeImage','predictedImage','observedImage'])$(id).hidden=true;
    badge('frameStatus','等待仿真帧');badge('connection','连接中');renderRuns();renderSummary();renderEvents();
    const run=runs.find(r=>r.run_id===runId);setText($('runTitle'),run?.run_path);setText($('runSubtitle'),`run ${runId} · 本地实验记录`);
    const locationURL=new URL(location.href);locationURL.searchParams.set('run',runId);history.replaceState(null,'',locationURL);
    try{
      const data=await getJSON(`/api/runs/${runId}`);if(token!==generation)return;
      summary=data.summary;setText($('artifactDetail'),pretty(data.run.manifest));renderSummary();updateArtifacts(data.artifacts);
      await loadEvents(token);if(token!==generation)return;
      stream=new EventSource(`/api/runs/${runId}/stream?after=${cursor}`);
      stream.onopen=()=>{if(token===generation)badge('connection','服务已连接','good');};
      stream.onerror=()=>{if(token===generation)badge('connection','正在重连','warn');};
      stream.addEventListener('snapshot',event=>{if(token!==generation)return;applySnapshot(JSON.parse(event.data));});
      notice('');tickFrame();
    }catch(error){if(token===generation)notice(error.message);}
  }
  async function loadEvents(token){const run=selected;const data=await getJSON(`/api/runs/${run}/events?after=${cursor}`);if(token===generation)applySnapshot(data);}
  function applySnapshot(data){if(data.reset){selectedEvent=null;prediction=null;selectedArtifact='';artifactVersion++;detailRequests.next();clearEvidenceViews($);}
    records=mergeEvents(records,data.events,data.reset);cursor=data.cursor;summary=data.summary;renderSummary();renderEvents();
    setText($('updatedAt'),`事件更新 ${new Date().toLocaleTimeString()} · 保留 ${records.length} 条`);
    setText($('sourceStatus'),`源代次 ${data.source.source_generation} · 诊断 ${data.source.error_count} 条${data.source.partial_line?' · 等待末行写完':''} · 界面最多保留 5000 条事件`);
    if(data.source.error_count)notice(`部分日志记录无法读取，最近诊断：${data.source.last_error?.kind}，第 ${data.source.last_error?.line} 行。原始日志仍被保留。`);
    if($('follow').checked&&data.events.length){selectedEvent=data.events[data.events.length-1];renderSelectedEvent();}
  }
  function renderSummary(){
    const s=summary,latest=s.latest||{},tasks=s.tasks||[],task=tasks[tasks.length-1]||{};
    const ended=tasks.filter(t=>!['running','incomplete'].includes(t.status)),success=ended.filter(t=>t.status==='succeeded');
    setText($('taskMetric'),tasks.length?`${success.length} / ${ended.length}`:'—');setText($('taskHint'),`${tasks.filter(t=>['running','incomplete'].includes(t.status)).length} 个运行中或无终态记录`);
    const latency=s.planning_latency_ms||{};setText($('latencyMetric'),latency.n?`${format(latency.p50,1)} / ${format(latency.p95,1)}`:'—');setText($('latencyHint'),`毫秒 · ${latency.n||0} 次记录`);
    const counts=s.event_counts||{};setText($('replanMetric'),`${counts.plan||0} / ${counts.replan_requested||0}`);
    const err=s.prediction_errors||{};const metric=['state_error','position_error_m','rmse_rad','frame_mse'].find(k=>err[k]?.n);
    setText($('errorMetric'),metric?format(err[metric].mean,5):'—');setText($('errorTitle'),metric?`预测误差 · ${metric}`:'预测误差');setText($('errorHint'),metric?`平均值 · ${err[metric].n} 条记录`:'未记录误差');
    const state=currentTaskState(latest,task),start=latest.task_started||{};
    const lastPlan=latest.plan||{},plan=lastPlan.task_id&&lastPlan.task_id!==task.task_id?{}:lastPlan;
    const ev=plan.evidence||plan.planning_evidence||{},predictionReport=plan.prediction||{};
    const status=task.status||state.status;badge('taskStatus',statusNames[status]||status||'未记录',status==='succeeded'?'good':status==='running'?'':'warn');
    const currentStart=start.task_id===task.task_id?start:{};
    pairs($('agentState'),[['任务目标',state.task_goal||currentStart.goal||task.goal],['当前子目标',state.active_node||ev.target_stage||latest.mission_goal?.goal],['已完成子目标',state.completed_subgoals],['可用技能',state.callable_skills||task.callable_skills],['当前候选技能',state.current_candidates],['执行 / 预算',`${format(state.executed_cycles??task.cycles)} / ${format(task.max_cycles)}`],['剩余预算',state.budget_remaining],['恢复尝试 / 已验证成功',`${format(state.recoveries)} / ${format(state.recoveries_succeeded)}`],['节点：状态 / 动作数 / 保持计数 / 恢复数',taskGraphRows(state)],['最近失败原因',state.failure_reason]]);
    $('taskNodeRows').replaceChildren();
    taskGraphRows(state).forEach(row=>cells($('taskNodeRows'),row));
    if(!taskGraphRows(state).length)cells($('taskNodeRows'),['当前日志未记录任务图','—','—','—','—']);
    pairs($('modelState'),[['基线',ev.baseline||start.baseline||latest.task_result?.baseline],['模型版本',plan.model_version||latest.task_result?.model_version],['预测跨度',plan.prefix_length??predictionReport.horizon_steps??plan.prediction_horizon_s],['不确定性语义',predictionReport.uncertainty_kind||((ev.baseline==='A3')?'校准误差界':undefined)],['当前动作',plan.actions||plan.action],['所选候选',ev.selected_candidate]]);
    setText($('recentFeedback'),pretty(state.recent_feedback||latest.feedback||latest.prediction_residual));
    setText($('robotState'),pretty(latest.observation?.state||latest.executed_transition?.after||latest.goal_result?.final_state||latest.feedback?.observed_state||state.recent_feedback?.observed_state));
    setText($('planDetails'),pretty(plan));$('candidateRows').replaceChildren();
    candidateViews(plan).forEach(c=>cells($('candidateRows'),[c.id,c.skill,c.score,c.prefix,c.chosen?'已选中':c.rejection||'候选'],c.chosen?'selected':''));
    if(!ev.candidates?.length)cells($('candidateRows'),['未记录候选明细','—','—','—','—']);setText($('candidateHint'),ev.candidates?`${ev.candidates.length} 个候选 · 分数越低越优（按当前规划器记录）`:'详见原始规划证据');
    const last=records[records.length-1]?.event,flowIndex={observation:0,task_started:0,language_planning:1,plan:2,execution_started:3,command_sent:3,feedback:4,prediction_residual:4,task_result:4}[last];
    $('flow').querySelectorAll('span').forEach((n,i)=>n.classList.toggle('active',i===flowIndex));
    plot($('prefixChart'),records.filter(r=>r.event==='plan'&&Number.isFinite(r.payload.prefix_length)).map(r=>r.payload.prefix_length),'动作步');
    const errorKey=metric||'state_error';setText($('errorChartTitle'),`预测残差 · ${errorKey}`);plot($('errorChart'),records.map(r=>r.payload[errorKey]).filter(Number.isFinite),errorKey);
  }
  function renderEvents(){
    const kinds=[...new Set(records.map(r=>r.event))];const old=$('eventFilter').value;$('eventFilter').replaceChildren(element('option','全部事件'));$('eventFilter').firstChild.value='';
    kinds.forEach(k=>{const option=element('option',names[k]||k);option.value=k;$('eventFilter').append(option);});$('eventFilter').value=old;
    const kind=$('eventFilter').value,episode=$('episodeFilter').value,step=$('stepFilter').value,query=$('eventSearch').value.toLowerCase();
    const filtered=records.filter(r=>{const p=r.payload;return (!kind||r.event===kind)&&(!episode||String(p.episode_id||'').includes(episode))&&(!step||String(p.step_id??p.observation_step)===step)&&(!query||JSON.stringify(p).toLowerCase().includes(query));});
    $('eventRows').replaceChildren();filtered.slice(-250).reverse().forEach(row=>{const p=row.payload;const tr=cells($('eventRows'),[row.cursor,names[row.event]||row.event,p.step_id??p.observation_step,row.wall_time_utc?.slice(11,23)||p.sim_time_s,p.task_id||p.episode_id],'event-row'+(selectedEvent?.cursor===row.cursor?' selected':''));tr.tabIndex=0;const pick=()=>{selectedEvent=row;renderSelectedEvent();};tr.onclick=pick;tr.onkeydown=e=>{if(e.key==='Enter')pick();};});setText($('eventCount'),`${filtered.length} / ${records.length} 条 · 展示最近 250 条`);
  }
  function renderSelectedEvent(){if(!selectedEvent)return;setText($('eventDetailTitle'),`${names[selectedEvent.event]||selectedEvent.event} · #${selectedEvent.cursor}`);setText($('eventDetail'),pretty(selectedEvent.raw));const artifact=selectedEvent.payload.artifact;
    $('showEventEvidence').hidden=!artifact;$('showEventEvidence').onclick=()=>{page('evidence');selectPrediction(artifact);};
    if(artifact&&$('follow').checked&&selectedArtifact!==artifact)selectPrediction(artifact);renderFrameAlignment();
  }
  async function tickFrame(){if(!selected||frameBusy)return;frameBusy=true;const token=generation,run=selected;
    try{const live=await getJSON(`/api/live/${run}/status`);if(token!==generation)return;const states={live:'实时观测',stale:'画面已过期',stopped:'仿真源已停止',unavailable:'未接入实时帧'};badge('frameStatus',states[live.state]||live.state,live.state==='live'?'good':'warn');
      pairs($('frameMeta'),[['相机',live.camera],['仿真时间 / s',live.sim_time_s],['回合',live.episode_id?.slice(0,12)],['步骤',live.step_id],['帧龄 / s',live.age_s],['覆盖旧帧',live.dropped_frames]],true);
      if(live.captured_at_utc&&currentFrame?.captured_at_utc!==live.captured_at_utc){const response=await fetch(`/api/live/${run}/frame.png`,{cache:'no-store'});if(!response.ok)throw new Error('帧暂不可用');const blob=await response.blob();if(token!==generation)return;
        if(frameURL)URL.revokeObjectURL(frameURL);frameURL=URL.createObjectURL(blob);$('liveFrame').src=frameURL;$('liveFrame').hidden=false;$('framePlaceholder').hidden=true;
        currentFrame={...live,episode_id:response.headers.get('X-Episode-ID'),step_id:Number(response.headers.get('X-Step-ID')),
          sim_time_s:Number(response.headers.get('X-Sim-Time-S')),captured_at_utc:response.headers.get('X-Captured-At-UTC')};renderFrameAlignment();}
    }catch(error){if(token===generation)badge('frameStatus','帧服务暂不可用','warn');}finally{frameBusy=false;}
  }
  function renderFrameAlignment(){const p=selectedEvent?.payload||{};const labels={exact:'画面与所选事件的回合 / 步号 / 仿真时间一致',same_step:'回合 / 步号一致，但缺少仿真时间，无法精确对齐',different_time:'回合 / 步号一致，仿真时间不同（动作段内或空闲画面）',different_episode:'画面与所选事件来自不同回合',different_step:'同一回合，画面与所选事件步号不同',unknown:'缺少回合或步号，无法精确对齐'};setText($('frameAlignment'),labels[alignment(currentFrame,p)]);}
  function updateArtifacts(list){artifacts=list;const predictions=list.filter(r=>r.kind==='prediction');$('predictionSelect').replaceChildren(element('option','选择预测工件'));$('predictionSelect').firstChild.value='';predictions.forEach(a=>{const option=element('option',a.relative_path);option.value=a.relative_path;$('predictionSelect').append(option);});if(predictions.some(p=>p.relative_path===selectedArtifact))$('predictionSelect').value=selectedArtifact;
    $('artifactList').replaceChildren();list.forEach(a=>{const button=element('button',undefined,'artifact-button');button.append(element('span',a.relative_path),element('small',`${a.kind} · ${format(a.size_bytes/1024,1)} KB`));button.onclick=()=>openArtifact(a);$('artifactList').append(button);});
  }
  const artifactURL=(path)=>`/api/runs/${selected}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`;
  async function selectPrediction(path){if(!path)return;selectedArtifact=path;const token=generation,version=++artifactVersion;try{const info=await getJSON(artifactURL(path)+'?info=1');if(token!==generation||version!==artifactVersion)return;prediction=info;$('predictionSelect').value=path;$('predictionStep').max=Math.max(0,info.frame_count-1);$('predictionStep').value=Math.max(0,info.frame_count-1);renderPrediction();}catch(error){if(token===generation)notice(error.message);}}
  function renderPrediction(){if(!prediction)return;const index=Number($('predictionStep').value);const info=prediction;const base=artifactURL(selectedArtifact);for(const [key,id] of [['before_rgb','beforeImage'],['predicted_rgb','predictedImage'],['observed_rgb','observedImage']]){const image=$(id);image.hidden=!info.available_images.includes(key);if(!image.hidden)image.src=base+`?key=${key}&index=${key==='predicted_rgb'?index:0}`;}
    setText($('predictionCaption'),`模型预测 · 第 ${index+1} 步`);setText($('predictionStepLabel'),`${index+1} / ${info.frame_count}`);setText($('predictionAlignment'),info.aligned?`回合 ${info.episode_id} · 步 ${info.before_step} → ${info.after_step} · 决策 ${info.decision_id}`:'旧工件未记录回合 / 步号对应关系；按工件内容浏览');
    const predicted=info.predicted_state?.[index],observed=index===info.frame_count-1?info.observed_state:null,bounds=info.error_bounds?.[index];stateTable(predicted,observed,bounds);setText($('uncertaintyNote'),index===info.frame_count-1?(bounds?'校准误差界仅在该模型与校准适用条件内解释。':'该工件没有校准误差界；不将预测误差或成员分歧换算为失败概率。'):'实际图像仅记录终端帧；中间预测步没有对应的实测状态，误差列留空。');
  }
  function stateTable(predicted,observed,bounds){$('stateRows').replaceChildren();if(!Array.isArray(predicted)){cells($('stateRows'),['未记录','—','—','—','—']);return;}const run=runs.find(r=>r.run_id===selected);const order=run?.manifest?.semantics?.state_order||[];predicted.forEach((value,i)=>cells($('stateRows'),[order[i]||`维度 ${i}`,value,observed?.[i],Number.isFinite(observed?.[i])?Math.abs(value-observed[i]):null,bounds?.[i]]));}
  async function openArtifact(artifact){if(artifact.kind==='prediction'){page('evidence');return selectPrediction(artifact.relative_path);}if(artifact.kind==='events'){page('history');return;}const token=generation,ticket=detailRequests.next();setText($('artifactTitle'),artifact.relative_path);setText($('artifactDetail'),'读取中');$('trainingChart').hidden=true;try{if(['json','manifest'].includes(artifact.kind)){const value=await getJSON(artifactURL(artifact.relative_path));if(token!==generation||!detailRequests.isCurrent(ticket))return;setText($('artifactDetail'),pretty(value));const history=Array.isArray(value)?value:value.history;$('trainingChart').hidden=!Array.isArray(history);if(Array.isArray(history))plot($('trainingChart'),history.map(row=>row.validation_loss).filter(Number.isFinite),'验证损失');}else if(artifact.kind==='image'){setText($('artifactDetail'),'图像工件可在浏览器单独查看');window.open(artifactURL(artifact.relative_path),'_blank','noopener');}}catch(error){if(token===generation&&detailRequests.isCurrent(ticket))notice(error.message);}}
  function plot(container,values,unit){container.replaceChildren();if(!values.length){setText(container,'暂无对应指标记录');return;}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 500 150');const create=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.append(n);return n;};const low=Math.min(...values),high=Math.max(...values),range=high-low||1;create('line',{x1:45,y1:122,x2:488,y2:122,stroke:'#354358'});create('polyline',{points:values.map((v,i)=>`${45+(i/Math.max(1,values.length-1))*440},${110-(v-low)/range*90}`).join(' '),fill:'none',stroke:'#8eb8ff','stroke-width':2});create('text',{x:4,y:22,fill:'#95a3b6','font-size':10},format(high,4));create('text',{x:4,y:115,fill:'#95a3b6','font-size':10},format(low,4));create('text',{x:45,y:144,fill:'#95a3b6','font-size':10},`顺序记录 · n=${values.length} · ${unit}`);container.append(svg);}
  async function compare(){const ids=[$('compareA').value,$('compareB').value];if(ids.some(id=>!id))return;try{const data=await Promise.all(ids.map(id=>getJSON(`/api/runs/${id}`)));setText($('compareNameA'),data[0].run.run_path);setText($('compareNameB'),data[1].run.run_path);$('compareRows').replaceChildren();const lines=[['代码提交',d=>d.run.git_commit],['种子',d=>d.run.seed],['事件窗口大小',d=>d.summary.retained_events],['任务状态计数',d=>d.summary.task_status_counts],['规划时延 p50 / ms',d=>d.summary.planning_latency_ms.p50],['规划时延 p95 / ms',d=>d.summary.planning_latency_ms.p95],['显式重规划请求',d=>d.summary.event_counts.replan_requested],['位置误差均值 / m',d=>d.summary.prediction_errors.position_error_m.mean],['关节 RMSE 均值 / rad',d=>d.summary.prediction_errors.rmse_rad.mean],['状态误差均值 / 原状态单位',d=>d.summary.prediction_errors.state_error.mean],['数据来源',d=>d.run.manifest.data_source||d.run.manifest.inputs]];lines.forEach(([label,fn])=>cells($('compareRows'),[label,fn(data[0]),fn(data[1])]));}catch(error){notice(error.message);}}
  $('refreshRuns').onclick=refreshRuns;$('runSearch').oninput=renderRuns;
  for(const id of ['eventFilter','episodeFilter','stepFilter','eventSearch'])$(id).oninput=renderEvents;
  $('predictionSelect').onchange=()=>selectPrediction($('predictionSelect').value);$('predictionStep').oninput=renderPrediction;$('compareRefresh').onclick=compare;
  setInterval(tickFrame,400);setInterval(refreshRuns,15000);
  setInterval(async()=>{if(!selected)return;const token=generation;try{const data=await getJSON(`/api/runs/${selected}/artifacts`);if(token===generation)updateArtifacts(data.artifacts);}catch(_){}},3000);
  window.addEventListener('beforeunload',()=>{stream?.close();if(frameURL)URL.revokeObjectURL(frameURL);});refreshRuns();
}
