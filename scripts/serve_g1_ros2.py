"""Own a G1 MuJoCo session in a ROS2 process; paused between bounded steps."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', default='/wmal/g1/session')
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args(argv)
    import rclpy
    from wmal.envs.indoor_scene import IndoorScene
    from wmal.envs.g1_session import G1MuJoCoSession
    from wmal.communication.g1_session_protocol import G1SessionOwner, scene_digest
    from wmal.communication.g1_ros2 import create_g1_session_node
    scene = IndoorScene()
    owner = G1SessionOwner(lambda: G1MuJoCoSession(viewer=args.viewer, realtime=False, scene=scene),
                           scene_digest(scene))
    node = None
    rclpy.init()
    try:
        node = create_g1_session_node(owner, args.service)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        owner.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
