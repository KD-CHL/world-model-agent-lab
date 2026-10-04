const assert = require('node:assert/strict');
const ui = require('../src/wmal/monitor/static/monitor.js');
assert.equal(ui.format(null), '未记录');
assert.equal(ui.format(NaN), '未记录');
assert.equal(ui.format(.25), '0.250');
assert.equal(ui.alignment({episode_id:'e1', step_id:4}, {episode_id:'e1', step_id:4}), 'same_step');
assert.equal(ui.alignment({episode_id:'e1', step_id:4,sim_time_s:1.2}, {episode_id:'e1', step_id:4,sim_time_s:1.2}), 'exact');
assert.equal(ui.alignment({episode_id:'e1', step_id:4,sim_time_s:1.4}, {episode_id:'e1', step_id:4,sim_time_s:1.2}), 'different_time');
assert.equal(ui.alignment({episode_id:'e2', step_id:4}, {episode_id:'e1', step_id:4}), 'different_episode');
assert.equal(ui.alignment({step_id:4}, {episode_id:'e1', step_id:4}), 'unknown');
assert.deepEqual(ui.mergeEvents([{cursor:1,event:'plan'}], [{cursor:1,event:'plan'}, {cursor:2,event:'feedback'}]).map(r=>r.cursor), [1,2]);
assert.deepEqual(ui.mergeEvents([{cursor:1}], [{cursor:3}], true), [{cursor:3}]);
const element = {textContent: ''};
ui.setText(element, '<img src=x onerror=alert(1)>');
assert.equal(element.textContent, '<img src=x onerror=alert(1)>');
assert.equal(typeof ui.RequestSequence, 'function');
const requests = new ui.RequestSequence();
const slow=requests.next(), fast=requests.next();
assert.equal(requests.isCurrent(slow), false);
assert.equal(requests.isCurrent(fast), true);
assert.equal(typeof ui.currentTaskState, 'function');
assert.deepEqual(ui.currentTaskState({task_result:{state:{status:'succeeded',task_goal:[1]}}},
                                    {task_id:'legacy-1',status:'incomplete'}), {});
assert.equal(typeof ui.candidateViews, 'function', 'Both visual and floating-G1 evidence must be projected');
assert.deepEqual(ui.candidateViews({planning_evidence:{selected_candidate:2,candidates:[
  {candidate_id:2,cost:.3,rejection:null},{candidate_id:3,cost:null,rejection:'predicted_constraint'}]}}), [
    {id:2,skill:null,score:.3,prefix:null,chosen:true,rejection:null},
    {id:3,skill:null,score:null,prefix:null,chosen:false,rejection:'predicted_constraint'}]);
assert.deepEqual(ui.candidateViews({evidence:{selected_candidate:0,candidates:[
  {candidate:0,skill:'arm_delta',score:.2,trusted_prefix:2}]}}), [
    {id:0,skill:'arm_delta',score:.2,prefix:2,chosen:true,rejection:null}]);
assert.equal(typeof ui.clearEvidenceViews, 'function', 'Run switches must clear previously displayed evidence');
const retained = new Map();
for (const id of ['eventDetail','eventDetailTitle','predictionAlignment','predictionStepLabel',
                 'artifactDetail','artifactTitle','frameAlignment','frameMeta','stateRows']) {
  retained.set(id, {textContent:'previous run evidence', hidden:false});
}
const getElement = id => retained.get(id);
ui.clearEvidenceViews(getElement);
for (const el of retained.values()) assert.notEqual(el.textContent, 'previous run evidence');
console.log('UI behavior: missing values, alignment, reconnect de-duplication, source reset and text-only rendering passed');
assert.equal(typeof ui.taskGraphRows, 'function');
assert.deepEqual(ui.taskGraphRows({nodes:{hold_A:{status:'running',actions:3,hold_count:2,recoveries:1}}}),
                [['hold_A','running',3,2,1]]);
assert.deepEqual(ui.taskGraphRows({}), []);

// Execute the actual rendering function, not just projection helpers.
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../src/wmal/monitor/static/monitor.js'),'utf8');
const renderBody = source.slice(source.indexOf('  function renderSummary(){'),source.indexOf('  function renderEvents(){'));
function renderedSummary(summary) {
  const nodes=new Map(), pairsRendered={};
  const $=id=>{
    if(!nodes.has(id))nodes.set(id,{id,textContent:'',replaceChildren(){},querySelectorAll(){return [];}});
    return nodes.get(id);
  };
  const context={...ui,summary,records:[],$,statusNames:{},
    badge:(id,text)=>ui.setText($(id),text),
    pairs:(node,rows)=>{pairsRendered[node.id]=Object.fromEntries(rows);},cells(){},plot(){},
    pretty:value=>JSON.stringify(value??'未记录',null,2)};
  vm.runInNewContext(renderBody+'\nrenderSummary();',context);
  return {nodes,pairs:pairsRendered};
}
const mixed=renderedSummary({tasks:[{task_id:'A',status:'succeeded'},{task_id:'B',status:'canceled',cycles:0}],
  latest:{task_started:{task_id:'B',goal:[.35,.87]},
    feedback:{task_id:'A',observed_state:[.43,.95,.43,.95]},
    observation:{task_id:'A',state:[9,9]},task_result:{task_id:'A',baseline:'A3',model_version:'old'},
    agent_state:{task_id:'B',state:{status:'canceled',executed_cycles:0}}}});
assert.equal(mixed.nodes.get('robotState').textContent,JSON.stringify('未记录'));
assert.equal(mixed.nodes.get('recentFeedback').textContent,JSON.stringify('未记录'));
assert.equal(mixed.pairs.modelState['模型版本'],undefined);
assert.equal(mixed.pairs.modelState['基线'],undefined);
const legacy=renderedSummary({tasks:[{task_id:'legacy',status:'failed'}],latest:{
  task_started:{task_id:'legacy'},mission_validated:{task_id:'legacy',goals:[{x:1,y:2}]},
  planning_event:{task_id:'legacy',reason:'obstacle_blocks_goal'}}});
assert.equal(JSON.stringify(legacy.pairs.agentState['任务目标']),JSON.stringify([{x:1,y:2}]));
assert.equal(legacy.pairs.agentState['最近失败原因'],'obstacle_blocks_goal');
