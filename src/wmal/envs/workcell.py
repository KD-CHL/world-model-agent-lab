"""Dataset-inspired tabletop worlds; geometry is procedural, not a scan replica."""
import numpy as np

from wmal.envs.indoor_scene import IndoorScene, Box
from wmal.envs.g1_locomotion import MODEL_XML, JOINT_NAMES, DEFAULT_JOINT_POS, KP, KD, EFFORT_LIMIT

TASKS = {
    'stack_block': 'Stack_Block', 'bag_insert': 'Bag_Insert',
    'erase_board': 'Erase_Board', 'clean_table': 'Clean_Table',
    'pack_pencilbox': 'Pack_PencilBox', 'pour_medicine': 'Pour_Medicine',
    'pack_pingpong': 'Pack_PingPong', 'prepare_fruit': 'Prepare_Fruit',
    'organize_tools': 'Organize_Tools', 'fold_towel': 'Fold_Towel',
    'wipe_table': 'Wipe_Table', 'dualrobot_clean_table': 'DualRobot_Clean_Table',
}


class WorkcellScene(IndoorScene):
    table_top = .78

    def __init__(self, task, seed=0):
        super().__init__()
        if task not in TASKS:
            raise ValueError('Unknown workcell task: ' + task)
        self.task, self.seed = task, seed
        self.boxes = tuple(b for b in self.boxes if b.name != 'island') + (
            Box('worktable', .85, 0., .45, .65, self.table_top),)
        self.objects = []
        self.regions = []

    def populate(self, spec, mj):
        # Reset descriptors on each build so the scene object is reusable.
        self.objects, self.regions = [], []
        room = IndoorScene()
        room.boxes = tuple(b for b in self.boxes if b.name != 'worktable')
        room.populate(spec, mj)
        self.spec, self.mj = spec, mj
        self.geom(spec.worldbody,'workcell_floor',[0,0,0],[0,0,.05],[.72,.75,.78,1],'plane')
        self.geom(spec.worldbody, 'table_top', [.85,0.,.75], [.45,.65,.03], [.65,.48,.3,1])
        for x in (.46, 1.24):
            for y in (-.59, .59):
                self.geom(spec.worldbody, f'leg_{x}_{y}', [x,y,.36], [.035,.035,.36], [.25,.28,.3,1])
        spec.worldbody.add_camera(name='workcell_overview', pos=[2.1,-2.1,2.3],
                                  xyaxes=[.86,.51,0.,-.28,.47,.84], fovy=48)
        spec.worldbody.add_camera(name='workcell_top', pos=[.85,0.,2.4],
                                  xyaxes=[0.,1.,0.,-1.,0.,0.], fovy=55)
        # Robot-facing RGB viewpoint; intrinsics are synthetic and uncalibrated.
        spec.worldbody.add_camera(name='workcell_front', pos=[.05,0.,1.35],
                                  xyaxes=[0.,-1.,0.,.58,0.,.82], fovy=70)
        rng = np.random.default_rng(self.seed)
        def item(name, x, y, size, color, kind='box', mass=.08):
            return self.object(name, x+float(rng.uniform(-.008,.008)), y,
                               size, color, kind, mass)
        red, blue, green = [ .85,.18,.12,1], [.15,.4,.85,1], [.18,.7,.35,1]
        if self.task == 'stack_block':
            for i, color in enumerate((red,blue,green)):
                item(f'block_{i}', .62, -.28+i*.25, [.035]*3, color)
            self.region('stack_target', 1.02,0.,.1,.1)
        elif self.task == 'bag_insert':
            self.container('bag_rigid_proxy', 1.,.18,.14,.15,.2, [.55,.4,.2,1])
            item('package', .6,-.25,[.06,.045,.04],blue)
        elif self.task == 'erase_board':
            self.geom(spec.worldbody,'board',[1.18,0.,1.06],[.018,.43,.28],[.94,.96,.94,1])
            for i in range(5):
                spec.worldbody.add_site(name=f'board_mark_{i}',pos=[1.159,-.3+i*.15,1.08],
                                       type=mj.mjtGeom.mjGEOM_BOX,size=[.001,.035,.01],rgba=blue)
            item('eraser',.62,0.,[.04,.07,.02],red)
        elif self.task in ('clean_table','dualrobot_clean_table'):
            self.container('collection_bin',1.,.38,.14,.15,.13,green)
            for i in range(5):
                item(f'clutter_{i}',.58+(i%2)*.23,-.38+(i//2)*.18,[.025,.035,.02],blue)
        elif self.task == 'pack_pencilbox':
            self.container('pencilbox',1.,.15,.12,.19,.07,blue)
            for i in range(3):
                item(f'pencil_{i}',.57+i*.08,-.3,[.009,.085,.009],red)
        elif self.task == 'pour_medicine':
            self.container('bottle',.63,-.22,.055,.055,.14,blue, dynamic=True)
            self.container('medicine_cup',1.,.22,.10,.10,.08,green)
            for i in range(6):
                body = item(f'pill_{i}',.61+(i%2)*.025,-.245+(i//2)*.023,[.008]*3,red,'sphere',.002)
                body.pos[2] += .016
        elif self.task == 'pack_pingpong':
            self.container('ball_box',1.,.24,.15,.17,.08,green)
            for i in range(3):
                item(f'ball_{i}',.6,-.34+i*.16,[.02]*3,[1.,.88,.2,1],'sphere',.003)
        elif self.task == 'prepare_fruit':
            self.container('fruit_tray',1.,.24,.15,.17,.04,green)
            for i,color in enumerate((red,green,[1,.6,.05,1])):
                item(f'fruit_{i}',.62,-.32+i*.17,[.038]*3,color,'sphere',.12)
        elif self.task == 'organize_tools':
            for i in range(3):
                self.container(f'tool_slot_{i}',1.,-.34+i*.32,.12,.12,.035,blue)
                body = item(f'tool_{i}',.6,-.32+i*.3,[.07,.015,.015],red)
                self.geom(body,f'tool_head_{i}',[.065,0.,0.],[.02,.035,.02],[.5,.5,.55,1])
        elif self.task == 'fold_towel':
            cloth = mj.MjSpec.from_string('''<mujoco><worldbody>
              <flexcomp name="towel" type="grid" count="9 9 1" spacing=".04 .04 .04"
                pos=".85 0 .81" dim="2" mass=".08" radius=".01" rgba=".15 .65 .8 1">
                <edge damping=".01"/><elasticity young="1000" poisson=".3" thickness=".002"/>
              </flexcomp></worldbody></mujoco>''')
            spec.attach(cloth, prefix='cloth_', frame=spec.worldbody.add_frame())
            self.region('fold_target',1.13,0.,.08,.16)
        elif self.task == 'wipe_table':
            item('sponge',.6,-.3,[.055,.035,.018],[1.,.8,.15,1])
            for i in range(6):
                self.region(f'stain_{i}',.78+(i%2)*.22,-.3+(i//2)*.25,.045,.045)

    def geom(self, parent, name, pos, size, color, kind='box', mass=None):
        kwargs = dict(name=name, pos=pos, size=size, rgba=color,
                      type=getattr(self.mj.mjtGeom,'mjGEOM_'+kind.upper()), friction=[.7,.005,.001])
        if mass is not None:
            kwargs['mass'] = mass
        return parent.add_geom(**kwargs)

    def object(self, name, x, y, size, color, kind='box', mass=.08):
        body = self.spec.worldbody.add_body(name=name,pos=[x,y,self.table_top+size[2]+.003])
        body.add_freejoint(name=name+'_free')
        self.geom(body,name+'_geom',[0,0,0],size,color,kind,mass)
        self.objects.append(name)
        return body

    def container(self, name, x, y, hx, hy, height, color, dynamic=False):
        body = self.spec.worldbody.add_body(name=name,pos=[x,y,self.table_top+.006])
        if dynamic:
            body.add_freejoint(name=name+'_free')
            self.objects.append(name)
        t = .006
        self.geom(body,name+'_bottom',[0,0,0],[hx,hy,t],color,mass=.04)
        for label,pos,size in (
            ('left',[-hx,0,height/2],[t,hy,height/2]),
            ('right',[hx,0,height/2],[t,hy,height/2]),
            ('front',[0,-hy,height/2],[hx,t,height/2]),
            ('back',[0,hy,height/2],[hx,t,height/2])):
            self.geom(body,name+'_'+label,pos,size,color,mass=.015)
        self.regions.append({'name':name,'center':[x,y,self.table_top], 'half_size':[hx,hy], 'height':height})

    def region(self,name,x,y,hx,hy):
        self.spec.worldbody.add_site(name=name,pos=[x,y,self.table_top+.002],
                                    type=self.mj.mjtGeom.mjGEOM_BOX,size=[hx,hy,.001],rgba=[.2,.75,.3,.5])
        self.regions.append({'name':name,'center':[x,y,self.table_top], 'half_size':[hx,hy]})

    def metadata(self):
        return {'task':self.task,'seed':self.seed,
                'dataset_reference':'https://huggingface.co/datasets/unitreerobotics/G1_'+TASKS[self.task],
                'fidelity':'procedural task-inspired geometry; not dataset reconstruction',
                'robots':2 if self.task=='dualrobot_clean_table' else 1,
                'robot_base':'fixed', 'dynamic_objects':self.objects,'regions':self.regions,
                'cameras':['workcell_overview','workcell_top','workcell_front'],
                'limitations':['Existing G1 model has no Dex1 actuated gripper; no manipulation policy loaded',
                    'Bag is a rigid open container proxy; towel is an uncalibrated flexible grid',
                    'Wipe/erase marks are visual annotations, not contact-driven task rewards',
                    'Fruit is rigid; medicine uses rigid pellets; no cutting or fluid physics']}


def build_workcell(task, seed=0):
    import mujoco as mj
    scene = WorkcellScene(task,seed)
    spec = mj.MjSpec()
    spec.option.timestep = .002
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    scene.populate(spec,mj)
    count = 2 if task == 'dualrobot_clean_table' else 1
    for robot in range(count):
        child = mj.MjSpec.from_file(str(MODEL_XML))
        child.delete(child.joint('floating_base_joint'))
        child.body('pelvis').pos = [0.,0.,.8]
        for i,name in enumerate(JOINT_NAMES):
            actuator = child.add_actuator(name=name+'_hold',trntype=mj.mjtTrn.mjTRN_JOINT,target=name)
            actuator.set_to_position(float(KP[i]),float(KD[i]))
            actuator.forcelimited = True
            actuator.forcerange = [-float(EFFORT_LIMIT[i]),float(EFFORT_LIMIT[i])]
        frame = spec.worldbody.add_frame(pos=[0.,0.,0.] if robot==0 else [1.7,0.,0.],
                                         quat=[1.,0.,0.,0.] if robot==0 else [0.,0.,0.,1.])
        spec.attach(child,prefix=f'g1_{robot}_',frame=frame)
    model = spec.compile()
    data = mj.MjData(model)
    for robot in range(count):
        for i,name in enumerate(JOINT_NAMES):
            jid = mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'g1_{robot}_'+name)
            aid = mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,f'g1_{robot}_'+name+'_hold')
            data.qpos[model.jnt_qposadr[jid]] = DEFAULT_JOINT_POS[i]
            data.ctrl[aid] = DEFAULT_JOINT_POS[i]
    mj.mj_forward(model,data)
    return scene, model, data
