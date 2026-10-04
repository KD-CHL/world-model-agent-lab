"""Task graph projections are isolated by task identity and support legacy logs."""
import unittest
from wmal.monitor.state import summarize


class TaskMonitorTests(unittest.TestCase):
    def test_task_nodes_and_budget_projected_without_cross_task_inheritance(self):
        rows=[{'event':'task_started','payload':{'task_id':'a','graph':{'nodes':[{'node_id':'reach_A'}]}}},
              {'event':'agent_state','payload':{'task_id':'a','state':{'active_node':'hold_A',
                 'nodes':{'hold_A':{'status':'running','hold_count':2}},'recoveries':1,
                 'budget_remaining':{'actions':15}}}},
              {'event':'task_started','payload':{'task_id':'b','goal':[.35,.87]}}]
        summary=summarize(rows)
        self.assertEqual(summary['tasks'][0]['active_node'],'hold_A')
        self.assertEqual(summary['tasks'][0]['nodes']['hold_A']['hold_count'],2)
        self.assertEqual(summary['tasks'][0]['budget_remaining']['actions'],15)
        self.assertNotIn('nodes',summary['tasks'][1])

    def test_fault_latched_task_is_preserved_as_terminal(self):
        summary=summarize([{'event':'task_result','payload':{'task_id':'a',
            'state':{'status':'fault_latched','executed_cycles':1,'failure_reason':'missing_trace'}}}])
        self.assertEqual(summary['tasks'][0]['status'],'fault_latched')
        self.assertEqual(summary['tasks'][0]['cycles'],1)

    def test_current_task_evidence_excludes_other_explicit_and_legacy_tasks(self):
        rows=[{'event':'feedback','payload':{'observed_state':[9,9]}},
              {'event':'task_result','payload':{'status':'succeeded'}},
              {'event':'task_started','payload':{'task_id':'new','goal':[1,2]}}]
        summary=summarize(rows)
        self.assertNotIn('feedback',summary['current_task_latest'])
        self.assertEqual(summary['current_task_latest']['task_started']['task_id'],'new')
        legacy=summarize(rows[:2])
        self.assertEqual(legacy['current_task_latest']['feedback']['task_id'],'legacy-0')
        self.assertEqual(legacy['current_task_latest']['feedback']['observed_state'],[9,9])
        self.assertNotIn('task_id',rows[0]['payload'])
