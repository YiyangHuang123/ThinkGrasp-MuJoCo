import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as SciPyRotation

from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm import osc as robosuite_osc
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.control_utils import nullspace_torques as robosuite_nullspace_torques
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.transform_utils import convert_quat, quat2mat
from robosuite.utils.camera_utils import (
    get_real_depth_map,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)

from scene_bridge import (
    FIXED_PERCEPTION_CROP_XYXY,
    OFFICIAL_WORKSPACE_XY_BOUNDS,
    OFFICIAL_WORKSPACE_Z_LIMITS,
    SCENE_VALID_WORKSPACE_SCALE,
)


# Project-level OSC posture configuration.
# robosuite 1.5.1 hard-codes nullspace_torques(..., joint_kp=10)
# inside OperationalSpaceController.run_controller() without exposing
# joint_kp in the controller config. Keep the installed robosuite package
# untouched and override only the OSC module reference in this process.
OSC_NULLSPACE_JOINT_KP = 30.0


def _thinkgrasp_nullspace_torques(
    mass_matrix,
    nullspace_matrix,
    initial_joint,
    joint_pos,
    joint_vel,
):
    return robosuite_nullspace_torques(
        mass_matrix,
        nullspace_matrix,
        initial_joint,
        joint_pos,
        joint_vel,
        joint_kp=OSC_NULLSPACE_JOINT_KP,
    )


robosuite_osc.nullspace_torques = _thinkgrasp_nullspace_torques



# ---------------------------------------------------------------------------
# Fixed GSO object sets for comparable multi-scene experiments.
#
# Scene01 is the original baseline scene. Scene02..Scene10 change only the
# five loaded GSO objects; robot, table, cameras, workspace, clutter-drop
# procedure, controller, bin, reward path, and task logic remain unchanged.
# ---------------------------------------------------------------------------
GSO_SCENE_OBJECT_SPECS = {
    "scene01": [
        {"name": "coffee_mug", "model_dir": "ACE_Coffee_Mug_Kristen_16_oz_cup"},
        {"name": "ecoforms_cup", "model_dir": "Ecoforms_Cup_B4_SAN"},
        {"name": "circo_holder", "model_dir": "Circo_Fish_Toothbrush_Holder_14995988"},
        {"name": "white_ramekin", "model_dir": "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup"},
        {"name": "ink_cartridge", "model_dir": "Canon_Pixma_Ink_Cartridge_8"},
    ],
    "scene02": [
        {"name": "black_bowl", "model_dir": "Now_Designs_Bowl_Akita_Black"},
        {"name": "nesquik_canister", "model_dir": "Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister"},
        {"name": "crayon_box", "model_dir": "Crayola_Bonus_64_Crayons"},
        {"name": "nikon_camera", "model_dir": "Nikon_1_AW1_w11275mm_Lens_Silver"},
        {"name": "bunny_racer", "model_dir": "BUNNY_RACER"},
    ],
    "scene03": [
        {"name": "white_cereal_bowl", "model_dir": "Threshold_Bead_Cereal_Bowl_White"},
        {"name": "latte_box", "model_dir": "Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g"},
        {"name": "green_speaker", "model_dir": "JBL_Charge_Speaker_portable_wireless_wired_Green"},
        {"name": "mario_figure", "model_dir": "Nintendo_Mario_Action_Figure"},
        {"name": "can_opener", "model_dir": "OXO_Soft_Works_Can_Opener_SnapLock"},
    ],
    "scene04": [
        {"name": "turquoise_bowl", "model_dir": "Room_Essentials_Bowl_Turquiose"},
        {"name": "fondant_box", "model_dir": "ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box"},
        {"name": "blue_bottle", "model_dir": "Perricone_MD_Cold_Plasma_Body"},
        {"name": "baby_car", "model_dir": "BABY_CAR"},
        {"name": "alarm_clock", "model_dir": "Crosley_Alarm_Clock_Vintage_Metal"},
    ],
    "scene05": [
        {"name": "yellow_blue_bowl", "model_dir": "Cole_Hardware_Bowl_Scirocco_YellowBlue"},
        {"name": "mocha_box", "model_dir": "Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct"},
        {"name": "gaming_mouse", "model_dir": "Razer_Abyssus_Ambidextrous_Gaming_Mouse"},
        {"name": "yoshi_figure", "model_dir": "Nintendo_Yoshi_Action_Figure"},
        {"name": "black_ink_box", "model_dir": "Office_Depot_Canon_PG21XL_Remanufactured_Ink_Cartridge_Black"},
    ],
    "scene06": [
        {"name": "quercetin_bottle", "model_dir": "Quercetin_500"},
        {"name": "cookie_candy_box", "model_dir": "Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box"},
        {"name": "fire_truck", "model_dir": "FIRE_TRUCK"},
        {"name": "moisturizer_jar", "model_dir": "Perricone_MD_Face_Finishing_Moisturizer_4_oz"},
        {"name": "pencil_case", "model_dir": "Big_Dot_Aqua_Pencil_Case"},
    ],
    "scene07": [
        {"name": "probiotic_bottle", "model_dir": "JarroDophilusFOS_Value_Size"},
        {"name": "snack_dispenser", "model_dir": "Snack_Catcher_Snack_Dispenser"},
        {"name": "rhino_figure", "model_dir": "Schleich_African_Black_Rhino"},
        {"name": "color_ink_box", "model_dir": "Brother_LC_1053PKS_Ink_Cartridge_CyanMagentaYellow_1pack"},
        {"name": "hard_drive", "model_dir": "Deskstar_Desk_Top_Hard_Drive_1_TB"},
    ],
    "scene08": [
        {"name": "creatine_bottle", "model_dir": "Twinlab_Nitric_Fuel"},
        {"name": "fujifilm_camera_box", "model_dir": "Fujifilm_instax_SHARE_SP1_10_photos"},
        {"name": "speed_boat", "model_dir": "SPEED_BOAT"},
        {"name": "face_moisturizer", "model_dir": "Perricone_MD_Face_Finishing_Moisturizer"},
        {"name": "peanut_butter_candy_box", "model_dir": "Nestle_Nips_Hard_Candy_Peanut_Butter"},
    ],
    "scene09": [
        {"name": "neck_cream_jar", "model_dir": "Perricone_MD_Firming_Neck_Therapy_Treatment"},
        {"name": "lion_figure", "model_dir": "Schleich_Lion_Action_Figure"},
        {"name": "soap_dish", "model_dir": "Threshold_Bamboo_Ceramic_Soap_Dish"},
        {"name": "pink_rubber_toy", "model_dir": "Kong_Puppy_Teething_Rubber_Small_Pink"},
        {"name": "bunny_racer", "model_dir": "BUNNY_RACER"},
    ],
    "scene10": [
        {"name": "borage_bottle", "model_dir": "Borage_GLA240Gamma_Tocopherol"},
        {"name": "toy_airplane", "model_dir": "TURBOPROP_AIRPLANE_WITH_PILOT"},
        {"name": "game_case", "model_dir": "Kid_Icarus_Uprising_Nintendo_3DS_Game"},
        {"name": "crocodile_toy", "model_dir": "My_First_Wiggle_Crocodile"},
        {"name": "cleanser_bottle", "model_dir": "Perricone_MD_Nutritive_Cleanser"},
    ],
}

# ---------------------------------------------------------------------------
# Validated Panda IK + JOINT_POSITION control configuration
# ---------------------------------------------------------------------------
#
# These parameters come from the standalone fixed-box validation in which the
# Panda completed:
#
#   confirmed initial pose
#   -> safe rest
#   -> over
#   -> 1 cm Cartesian descent
#   -> close
#   -> 1 cm Cartesian lift
#
# and successfully lifted the test box.
#
# The existing OSC implementation below remains untouched and available for
# rollback.
IK_CONFIRMED_INITIAL_JOINTS = np.array(
    [
        0.0,
        0.19634954,
        0.0,
        -2.61799388,
        0.0,
        2.94159265,
        0.78539816,
    ],
    dtype=np.float64,
)

IK_SAFE_REST_JOINTS = np.array(
    [
        0.0,
        0.222100209,
        0.0,
        -2.04897335,
        0.0,
        2.39832280,
        0.785398163,
    ],
    dtype=np.float64,
)

IK_QREF_SPEED_RAD_PER_SEC = 1.0
IK_JOINT_KP = 300.0
IK_JOINT_DAMPING_RATIO = 1.0
IK_CARTESIAN_WAYPOINT_SPACING = 0.01

# Execution-quality gate for non-strict IK solutions.
#
# residual_threshold=1e-5 remains unchanged and continues to define strict
# solver convergence. A non-converged multi-start result may still be
# executed, but only when its actual Cartesian errors are already small.
#
# This prevents a merely finite "best of several bad seeds" solution from
# being sent to JOINT_POSITION.
IK_BEST_EFFORT_MAX_POSITION_ERROR_M = 0.010
IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG = 2.0

# End-effector force-stop for grasp descent.
# The stopping metric is the sum of absolute values of the six-component
# end-effector wrench. Requiring consecutive threshold crossings avoids
# reacting to a one-step impact spike.
PYBULLET_STYLE_GRASP_FORCE_THRESHOLD = 15.0
GRASP_FORCE_STOP_CONSECUTIVE_STEPS = 5


def load_thinkgrasp_joint_position_controller_config():
    """Return the Panda JOINT_POSITION composite controller config."""

    config = load_composite_controller_config(
        controller="BASIC"
    )

    config["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": IK_JOINT_KP,
        "damping_ratio": IK_JOINT_DAMPING_RATIO,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }

    return config


class ThinkGraspMinimalEnv(ManipulationEnv):
    def __init__(
        self,
        robots="Panda",
        controller_configs=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=480,
        camera_widths=640,
        camera_depths=True,
        camera_segmentations="instance",
        hard_reset=True,
        scene_name="scene01",
        **kwargs,
    ):
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0.07, 0.0, 0.83))

        # Front / left / right oblique perception cameras around the clutter centre.
        # All three use matched distance, height, FOV, and approximately
        # 25-degree downward viewing geometry.
        self.left_oblique_camera_name = "left_oblique_25deg"
        self.right_oblique_camera_name = "right_oblique_25deg"
        self.front_oblique_camera_name = "front_oblique_25deg"

        self.front_oblique_camera_position = np.array(
            [0.87, 0.00, 1.26304662],
            dtype=np.float64,
        )

        self.left_oblique_camera_position = np.array(
            [0.07, 0.80, 1.26304662],
            dtype=np.float64,
        )
        self.right_oblique_camera_position = np.array(
            [0.07, -0.80, 1.26304662],
            dtype=np.float64,
        )

        self.oblique_camera_target = np.array(
            [0.07, 0.00, 0.89],
            dtype=np.float64,
        )
        self.oblique_camera_pitch_deg = 25.0

        # The perception workspace is resolved through
        # _get_perception_workspace_limits().

        # Initial clutter construction independently samples object XY
        # positions, then drops and settles objects sequentially inside a compact
        # MuJoCo drop rectangle. Invalid scenes are regenerated.
        self.object_drop_world_bounds = np.array(
            [
                [0.10, 0.18],
                [-0.06, 0.06],
            ],
            dtype=np.float64,
        )
        self.object_drop_height_above_table = 0.18

        # Google Scanned Objects used in the selected clutter scene.
        self.gso_models_root = (
            Path(__file__).resolve().parent
            / "assets"
            / "scanned_objects"
            / "models"
        )
        self.gso_adapter_root = (
            Path(__file__).resolve().parent
            / "assets"
            / "scanned_objects"
            / "robosuite_adapters"
        )

        self.scene_name = str(scene_name).strip().lower()

        if self.scene_name not in GSO_SCENE_OBJECT_SPECS:
            raise ValueError(
                f"Unknown scene_name {self.scene_name!r}. "
                f"Available scenes: {sorted(GSO_SCENE_OBJECT_SPECS)}"
            )

        self.gso_object_specs = [
            dict(object_spec)
            for object_spec in GSO_SCENE_OBJECT_SPECS[self.scene_name]
        ]

        self.object_drop_min_steps = 220
        self.object_drop_max_steps = 1400
        self.object_drop_check_interval = 20
        self.object_drop_stable_checks = 8
        self.object_drop_linear_speed_threshold = 0.015
        self.object_drop_angular_speed_threshold = 0.08

        # Reject reset scenes in which any object leaves the tabletop.
        self.object_clutter_max_retries = 50
        self.object_table_validity_margin_xy = 0.02
        self.object_table_validity_z_tolerance = 0.04

        # Fixed receiving bin on the robot's left side (world +Y).
        # The bin is outside the tabletop with its opening below table height.
        self.bin_center = np.array(
            [0.07, 0.40],
            dtype=np.float64,
        )
        self.bin_inner_half_size = np.array(
            [0.16, 0.14],
            dtype=np.float64,
        )
        self.bin_wall_thickness = 0.015
        self.bin_floor_top_z = 0.61
        self.bin_top_z = 0.81

        # Release position above the receiving bin.
        self.bin_release_position = np.array(
            [
                self.bin_center[0],
                self.bin_center[1] + 0.05,
                0.97,
            ],
            dtype=np.float64,
        )

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types="default",
            base_types="default",
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            control_freq=20,
            horizon=100,
            ignore_done=True,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            **kwargs,
        )

        # Build the runtime MuJoCo perception workspace.
        # Perception and GraspNet paths share this single 3D box.
        self.perception_workspace_limits = (
            self._get_perception_workspace_limits()
        )

        # robosuite stores render_camera as an instance attribute,
        # which shadows the custom render_camera() method below.
        self._robosuite_render_camera = self.render_camera
        del self.render_camera

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()

        robot_base_pos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(robot_base_pos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # Fixed top-view camera above the table centre, looking vertically down.
        # MuJoCo cameras look along local -Z, so the identity quaternion is used.
        topview_camera = ET.Element(
            "camera",
            attrib={
                "name": "topview",
                "mode": "fixed",
                "pos": "0.07 0 2.23",
                "quat": "1 0 0 0",
                "fovy": "32",
            },
        )
        arena.worldbody.append(topview_camera)

        # Front / left / right oblique cameras with matched viewing geometry.
        # All three use the same 0.80 m horizontal distance,
        # the same height, the same approximately 25-degree
        # downward viewing angle, and the same 45-degree FOV.
        #
        # Quaternion order in MuJoCo XML is w x y z.
        front_oblique_camera = ET.Element(
            "camera",
            attrib={
                "name": self.front_oblique_camera_name,
                "mode": "fixed",
                "pos": " ".join(
                    str(float(value))
                    for value in self.front_oblique_camera_position
                ),
                "quat": "0.59636781 -0.37992820 0.37992820 -0.59636781",
                "fovy": "45",
            },
        )
        arena.worldbody.append(front_oblique_camera)

        left_oblique_camera = ET.Element(
            "camera",
            attrib={
                "name": self.left_oblique_camera_name,
                "mode": "fixed",
                "pos": " ".join(
                    str(float(value))
                    for value in self.left_oblique_camera_position
                ),
                "quat": "0 0 0.53729961 0.84339145",
                "fovy": "45",
            },
        )
        arena.worldbody.append(left_oblique_camera)

        right_oblique_camera = ET.Element(
            "camera",
            attrib={
                "name": self.right_oblique_camera_name,
                "mode": "fixed",
                "pos": " ".join(
                    str(float(value))
                    for value in self.right_oblique_camera_position
                ),
                "quat": "0 0 -0.53729961 0.84339145",
                "fovy": "45",
            },
        )
        arena.worldbody.append(right_oblique_camera)

        # Fixed receiving bin on the left side of the table.
        # Built from one floor and four low collision walls.
        bin_half_x = float(self.bin_inner_half_size[0])
        bin_half_y = float(self.bin_inner_half_size[1])
        wall_t = float(self.bin_wall_thickness)
        floor_half_z = 0.01
        wall_half_z = (
            self.bin_top_z - self.bin_floor_top_z
        ) / 2.0
        wall_center_z = (
            self.bin_top_z + self.bin_floor_top_z
        ) / 2.0

        bin_geoms = [
            {
                "name": "left_bin_bottom",
                "pos": [
                    self.bin_center[0],
                    self.bin_center[1],
                    self.bin_floor_top_z - floor_half_z,
                ],
                "size": [
                    bin_half_x + wall_t,
                    bin_half_y + wall_t,
                    floor_half_z,
                ],
            },
            {
                "name": "left_bin_wall_near",
                "pos": [
                    self.bin_center[0],
                    self.bin_center[1] - bin_half_y - wall_t,
                    wall_center_z,
                ],
                "size": [
                    bin_half_x + wall_t,
                    wall_t,
                    wall_half_z,
                ],
            },
            {
                "name": "left_bin_wall_far",
                "pos": [
                    self.bin_center[0],
                    self.bin_center[1] + bin_half_y + wall_t,
                    wall_center_z,
                ],
                "size": [
                    bin_half_x + wall_t,
                    wall_t,
                    wall_half_z,
                ],
            },
            {
                "name": "left_bin_wall_robot_side",
                "pos": [
                    self.bin_center[0] - bin_half_x - wall_t,
                    self.bin_center[1],
                    wall_center_z,
                ],
                "size": [
                    wall_t,
                    bin_half_y,
                    wall_half_z,
                ],
            },
            {
                "name": "left_bin_wall_outer_side",
                "pos": [
                    self.bin_center[0] + bin_half_x + wall_t,
                    self.bin_center[1],
                    wall_center_z,
                ],
                "size": [
                    wall_t,
                    bin_half_y,
                    wall_half_z,
                ],
            },
        ]

        for geom_config in bin_geoms:
            bin_geom = ET.Element(
                "geom",
                attrib={
                    "name": geom_config["name"],
                    "type": "box",
                    "pos": " ".join(
                        str(float(value))
                        for value in geom_config["pos"]
                    ),
                    "size": " ".join(
                        str(float(value))
                        for value in geom_config["size"]
                    ),
                    "rgba": "0.16 0.18 0.22 1.0",
                    "friction": "1.0 0.005 0.0001",
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "1",
                },
            )
            arena.worldbody.append(bin_geom)

        self.objects = []

        for object_spec in self.gso_object_specs:
            adapter_xml = self._prepare_gso_xml_for_robosuite(
                model_dir_name=object_spec["model_dir"],
            )

            self.objects.append(
                MujocoXMLObject(
                    fname=str(adapter_xml),
                    name=object_spec["name"],
                    joints="default",
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
            )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    @staticmethod
    def _read_obj_vertex_bounds(obj_path):
        """Return min / max XYZ bounds from a Wavefront OBJ vertex list."""

        minimum = np.array(
            [np.inf, np.inf, np.inf],
            dtype=np.float64,
        )
        maximum = np.array(
            [-np.inf, -np.inf, -np.inf],
            dtype=np.float64,
        )
        vertex_count = 0

        with open(
            obj_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as obj_file:
            for line in obj_file:
                if not line.startswith("v "):
                    continue

                fields = line.split()

                if len(fields) < 4:
                    continue

                vertex = np.array(
                    [
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    ],
                    dtype=np.float64,
                )

                minimum = np.minimum(minimum, vertex)
                maximum = np.maximum(maximum, vertex)
                vertex_count += 1

        if vertex_count == 0:
            raise ValueError(
                f"No OBJ vertices found in {obj_path}"
            )

        return minimum, maximum

    def _prepare_gso_xml_for_robosuite(
        self,
        model_dir_name,
    ):
        """Create a robosuite-compatible wrapper around one GSO MJCF file.

        The downloaded GSO files use their own visual / collision geom groups
        and a single top-level body. MujocoXMLObject expects the standard
        robosuite XML-object structure. The source files are never modified:
        a small derived adapter XML is generated under robosuite_adapters/.
        """

        source_dir = (
            self.gso_models_root
            / model_dir_name
        )
        source_xml = source_dir / "model.xml"

        if not source_xml.exists():
            raise FileNotFoundError(
                f"Missing GSO model XML: {source_xml}"
            )

        tree = ET.parse(source_xml)
        source_root = tree.getroot()

        source_asset = source_root.find("asset")
        source_worldbody = source_root.find("worldbody")

        if source_asset is None or source_worldbody is None:
            raise ValueError(
                f"Invalid GSO MJCF structure: {source_xml}"
            )

        source_body = source_worldbody.find("body")

        if source_body is None:
            raise ValueError(
                f"No object body found in {source_xml}"
            )

        # Estimate the object's geometric extent from the high-resolution
        # visual mesh. These sites are metadata required by robosuite's XML
        # object interface; they do not change the GSO collision geometry.
        visual_mesh_file = None

        for mesh in source_asset.findall("mesh"):
            if mesh.get("name") == "model":
                visual_mesh_file = mesh.get("file")
                break

        if visual_mesh_file is None:
            raise ValueError(
                f"No visual mesh named 'model' found in {source_xml}"
            )

        visual_mesh_path = (
            source_dir
            / visual_mesh_file
        )

        minimum, maximum = self._read_obj_vertex_bounds(
            visual_mesh_path
        )
        horizontal_radius = float(
            max(
                abs(minimum[0]),
                abs(maximum[0]),
                abs(minimum[1]),
                abs(maximum[1]),
            )
        )

        adapter_root = ET.Element(
            "mujoco",
            attrib={"model": model_dir_name},
        )
        adapter_asset = ET.SubElement(
            adapter_root,
            "asset",
        )

        # Copy assets while rewriting referenced files relative to the
        # generated adapter XML directory, so the project remains portable
        # after moving the repository.
        adapter_dir = (
            self.gso_adapter_root
            / model_dir_name
        )

        for asset_element in list(source_asset):
            copied = ET.fromstring(
                ET.tostring(
                    asset_element,
                    encoding="unicode",
                )
            )

            file_attribute = copied.get("file")
            if file_attribute:
                resolved_file = (
                    source_dir
                    / file_attribute
                ).resolve()
                relative_file = Path(
                    os.path.relpath(
                        resolved_file,
                        start=adapter_dir.resolve(),
                    )
                )
                copied.set(
                    "file",
                    relative_file.as_posix(),
                )

            adapter_asset.append(copied)

        adapter_worldbody = ET.SubElement(
            adapter_root,
            "worldbody",
        )
        outer_body = ET.SubElement(
            adapter_worldbody,
            "body",
            attrib={"name": "wrapper"},
        )

        object_body = ET.fromstring(
            ET.tostring(
                source_body,
                encoding="unicode",
            )
        )
        object_body.set("name", "object")

        # robosuite XML objects use group 0 for collision and group 1 for
        # visual geoms. The GSO conversion used group 3 and group 2.
        for geom in object_body.iter("geom"):
            group = geom.get("group")

            if group == "2":
                geom.set("group", "1")
            elif group == "3":
                geom.set("group", "0")

        outer_body.append(object_body)

        def add_site(name, position):
            ET.SubElement(
                outer_body,
                "site",
                attrib={
                    "name": name,
                    "pos": " ".join(
                        str(float(value))
                        for value in position
                    ),
                    "size": "0.001",
                    "rgba": "0 0 0 0",
                },
            )

        add_site(
            "bottom_site",
            [0.0, 0.0, float(minimum[2])],
        )
        add_site(
            "top_site",
            [0.0, 0.0, float(maximum[2])],
        )
        add_site(
            "horizontal_radius_site",
            [horizontal_radius, 0.0, 0.0],
        )

        adapter_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        adapter_xml = adapter_dir / "model.xml"

        ET.ElementTree(adapter_root).write(
            adapter_xml,
            encoding="utf-8",
            xml_declaration=True,
        )

        return adapter_xml

    def _setup_references(self):
        super()._setup_references()

        self.object_body_ids = {
            obj.name: self.sim.model.body_name2id(obj.root_body)
            for obj in self.objects
        }

        self.body_id_to_name = {
            body_id: name
            for name, body_id in self.object_body_ids.items()
        }

        # Compatibility object-ID structure.
        self.obj_ids = {
            "fixed": [],
            "rigid": list(self.object_body_ids.values()),
        }

        # The language task and target object are supplied externally by
        # run_closed_loop.py after VLM target selection.
        self.lang_goal = ""
        self.target_obj_names = []
        self.target_obj_ids = []

    def set_language_task(
        self,
        goal,
        target_object_name,
    ):
        """Update the current externally supplied language task target."""

        target_object_name = str(
            target_object_name
        ).strip()

        if target_object_name not in self.object_body_ids:
            raise KeyError(
                "Unknown task target object "
                f"{target_object_name!r}. Available objects: "
                f"{sorted(self.object_body_ids)}"
            )

        self.lang_goal = str(goal).strip()
        self.target_obj_names = [target_object_name]
        self.target_obj_ids = [
            self.object_body_ids[target_object_name]
        ]

    def _reset_internal(self):
        super()._reset_internal()

        # Save the Panda reset posture before any object is dropped.
        # During clutter construction, raw self.sim.step() calls advance
        # physics without running robosuite's OSC controller. The saved
        # state is therefore restored around every raw physics step so the
        # robot remains fixed while the objects still fall and collide.
        self._capture_robot_initialization_state()

        try:
            # GSO clutter scene:
            # sample each object independently inside the approved perception
            # workspace, then drop and settle objects one by one.
            self._build_initial_clutter_by_center_drop()
        finally:
            self._restore_robot_initialization_state()
            self.sim.forward()

    def _capture_robot_initialization_state(self):
        """Save arm and gripper state for temporary reset-time freezing."""

        robot = self.robots[0]

        qpos_indexes = list(
            np.asarray(
                robot._ref_joint_pos_indexes,
                dtype=np.int64,
            ).reshape(-1)
        )
        qvel_indexes = list(
            np.asarray(
                robot._ref_joint_vel_indexes,
                dtype=np.int64,
            ).reshape(-1)
        )

        # The arm reference indexes do not necessarily contain the two
        # Panda finger joints, so include the gripper joints explicitly.
        gripper = getattr(robot, "gripper", None)

        if gripper is not None:
            for joint_name in getattr(gripper, "joints", []):
                joint_id = self.sim.model.joint_name2id(joint_name)

                qpos_address = int(
                    self.sim.model.jnt_qposadr[joint_id]
                )
                dof_address = int(
                    self.sim.model.jnt_dofadr[joint_id]
                )

                qpos_indexes.append(qpos_address)
                qvel_indexes.append(dof_address)

        self._initialization_robot_qpos_indexes = np.unique(
            np.asarray(qpos_indexes, dtype=np.int64)
        )
        self._initialization_robot_qvel_indexes = np.unique(
            np.asarray(qvel_indexes, dtype=np.int64)
        )

        self._initialization_robot_qpos = self.sim.data.qpos[
            self._initialization_robot_qpos_indexes
        ].copy()
        self._initialization_robot_qvel = self.sim.data.qvel[
            self._initialization_robot_qvel_indexes
        ].copy()

    def _restore_robot_initialization_state(self):
        """Restore the saved Panda state without changing object states."""

        self.sim.data.qpos[
            self._initialization_robot_qpos_indexes
        ] = self._initialization_robot_qpos

        # Keep the robot fully stationary during the temporary freeze.
        self.sim.data.qvel[
            self._initialization_robot_qvel_indexes
        ] = 0.0

        if hasattr(self.sim.data, "qacc"):
            self.sim.data.qacc[
                self._initialization_robot_qvel_indexes
            ] = 0.0

    def _random_drop_quaternion_wxyz(self):
        """Return a PyBullet-style random drop orientation.

        The original ThinkGrasp PyBullet environment samples all three Euler
        angles independently over [0, 2*pi). Keep the same sampling rule here
        and convert to MuJoCo quaternion order (w, x, y, z).
        """

        euler_xyz = self.rng.uniform(
            0.0,
            2.0 * np.pi,
            size=3,
        )

        quaternion_xyzw = (
            SciPyRotation.from_euler(
                "xyz",
                euler_xyz,
            ).as_quat()
        )

        return np.array(
            [
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            ],
            dtype=np.float64,
        )


    def _set_free_object_pose(
        self,
        obj,
        position,
        quaternion_wxyz,
    ):
        """Set one free object's pose and clear its residual velocity."""

        self.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate(
                [
                    np.asarray(position, dtype=np.float64),
                    np.asarray(quaternion_wxyz, dtype=np.float64),
                ]
            ),
        )

        joint_id = self.sim.model.joint_name2id(obj.joints[0])
        velocity_address = int(
            self.sim.model.jnt_dofadr[joint_id]
        )
        self.sim.data.qvel[
            velocity_address:velocity_address + 6
        ] = 0.0

    def _wait_for_dropped_object_to_settle(
        self,
        body_id,
    ):
        """Advance physics until the newly dropped object is stable."""

        stable_count = 0

        for step_index in range(self.object_drop_max_steps):
            # Raw MuJoCo stepping is required here so the newly introduced
            # object can fall naturally. Restore the Panda immediately before
            # and after that step, preventing gravity-driven arm collapse
            # while leaving every object state untouched.
            self._restore_robot_initialization_state()
            self.sim.forward()
            self.sim.step()
            self._restore_robot_initialization_state()
            self.sim.forward()

            if step_index + 1 < self.object_drop_min_steps:
                continue

            if (
                (step_index + 1)
                % self.object_drop_check_interval
                != 0
            ):
                continue

            body_velocity = np.asarray(
                self.sim.data.cvel[body_id],
                dtype=np.float64,
            )

            angular_speed = float(
                np.linalg.norm(body_velocity[:3])
            )
            linear_speed = float(
                np.linalg.norm(body_velocity[3:])
            )

            if (
                linear_speed
                <= self.object_drop_linear_speed_threshold
                and angular_speed
                <= self.object_drop_angular_speed_threshold
            ):
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= self.object_drop_stable_checks:
                return {
                    "settled": True,
                    "steps": step_index + 1,
                    "linear_speed": linear_speed,
                    "angular_speed": angular_speed,
                }

        body_velocity = np.asarray(
            self.sim.data.cvel[body_id],
            dtype=np.float64,
        )

        return {
            "settled": False,
            "steps": self.object_drop_max_steps,
            "linear_speed": float(
                np.linalg.norm(body_velocity[3:])
            ),
            "angular_speed": float(
                np.linalg.norm(body_velocity[:3])
            ),
        }

    def _park_all_clutter_objects(self):
        """Move all clutter objects away from the table before a retry."""

        for index, obj in enumerate(self.objects):
            parking_position = np.array(
                [
                    2.0 + 0.20 * index,
                    2.0,
                    0.30,
                ],
                dtype=np.float64,
            )

            self._set_free_object_pose(
                obj=obj,
                position=parking_position,
                quaternion_wxyz=np.array(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                ),
            )

        self.sim.forward()

    def _validate_clutter_scene(
        self,
        drop_results,
    ):
        """Validate the complete reset clutter scene.

        A scene is accepted only when:
            1. every object is still on the tabletop;
            2. every object's final body centre is inside the approved purple
               perception workspace;
            3. every sequential drop reported settled=True.

        Any failure regenerates the complete selected scene.
        """

        table_top_z = float(
            self.table_offset[2]
            + self.table_full_size[2] / 2.0
        )
        table_half_x = float(self.table_full_size[0]) / 2.0
        table_half_y = float(self.table_full_size[1]) / 2.0

        table_x_min = (
            float(self.table_offset[0])
            - table_half_x
            + float(self.object_table_validity_margin_xy)
        )
        table_x_max = (
            float(self.table_offset[0])
            + table_half_x
            - float(self.object_table_validity_margin_xy)
        )
        table_y_min = (
            float(self.table_offset[1])
            - table_half_y
            + float(self.object_table_validity_margin_xy)
        )
        table_y_max = (
            float(self.table_offset[1])
            + table_half_y
            - float(self.object_table_validity_margin_xy)
        )

        minimum_valid_z = (
            table_top_z
            - float(self.object_table_validity_z_tolerance)
        )

        perception_bounds = (
            self._get_perception_workspace_world_bounds()
        )

        # Scene-valid region: centred scaled copy of the runtime
        # workspace. This is only an initial clutter acceptance constraint;
        # it does not shrink the perception / GraspNet workspace.
        workspace_center = np.mean(
            perception_bounds,
            axis=1,
        )
        workspace_half_extent = (
            0.5
            * (
                perception_bounds[:, 1]
                - perception_bounds[:, 0]
            )
        )

        scene_valid_half_extent = (
            workspace_half_extent
            * float(SCENE_VALID_WORKSPACE_SCALE)
        )

        scene_valid_bounds = np.column_stack(
            [
                workspace_center - scene_valid_half_extent,
                workspace_center + scene_valid_half_extent,
            ]
        )

        settle_by_name = {
            result["object_name"]: bool(result["settled"])
            for result in drop_results
        }

        invalid_objects = []

        for obj in self.objects:
            body_id = self.object_body_ids[obj.name]
            position = np.asarray(
                self.sim.data.body_xpos[body_id],
                dtype=np.float64,
            ).copy()

            inside_table_xy = (
                table_x_min <= float(position[0]) <= table_x_max
                and table_y_min <= float(position[1]) <= table_y_max
            )
            high_enough = (
                float(position[2]) >= minimum_valid_z
            )

            inside_perception_xy = (
                scene_valid_bounds[0, 0]
                <= float(position[0])
                <= scene_valid_bounds[0, 1]
                and scene_valid_bounds[1, 0]
                <= float(position[1])
                <= scene_valid_bounds[1, 1]
            )

            settled = bool(
                settle_by_name.get(obj.name, False)
            )

            if not (
                inside_table_xy
                and high_enough
                and inside_perception_xy
                and settled
            ):
                invalid_objects.append(
                    {
                        "object_name": obj.name,
                        "position": position,
                        "inside_table_xy": bool(inside_table_xy),
                        "high_enough": bool(high_enough),
                        "inside_perception_xy": bool(
                            inside_perception_xy
                        ),
                        "settled": settled,
                    }
                )

        return {
            "valid": len(invalid_objects) == 0,
            "invalid_objects": invalid_objects,
            "perception_world_bounds": perception_bounds,
            "scene_valid_world_bounds": scene_valid_bounds,
            "scene_valid_workspace_scale": float(
                SCENE_VALID_WORKSPACE_SCALE
            ),
        }


    def _topview_pixel_to_table_xy(
        self,
        pixel_xy,
    ):
        """Project one topview pixel onto the tabletop plane."""

        pixel_xy = np.asarray(
            pixel_xy,
            dtype=np.float64,
        ).reshape(2)

        width = 640
        height = 640

        intrinsics = self.get_camera_intrinsics(
            camera_name="topview",
            width=width,
            height=height,
        )
        camera_to_world = self.get_camera_extrinsics(
            camera_name="topview",
        )

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        u = float(pixel_xy[0])
        v = float(pixel_xy[1])

        # Keep the same vertical pixel convention used by get_pointcloud().
        v_corrected = height - 1 - v

        ray_camera = np.array(
            [
                (u - cx) / fx,
                (v_corrected - cy) / fy,
                1.0,
            ],
            dtype=np.float64,
        )

        ray_world = (
            camera_to_world[:3, :3]
            @ ray_camera
        )
        camera_world = (
            camera_to_world[:3, 3]
        )

        table_top_z = float(
            self.table_offset[2]
            + self.table_full_size[2] / 2.0
        )

        if abs(float(ray_world[2])) < 1e-9:
            raise RuntimeError(
                "Topview ray is parallel to tabletop"
            )

        scale = (
            table_top_z
            - float(camera_world[2])
        ) / float(ray_world[2])

        if scale <= 0.0:
            raise RuntimeError(
                "Topview pixel projects behind the camera"
            )

        world_point = (
            camera_world
            + scale * ray_world
        )

        return world_point[:2].astype(
            np.float64
        )

    def _get_perception_workspace_world_bounds(self):
        """Return the configured tabletop world-XY workspace.

        The Panda-facing X-min margin is already included in
        OFFICIAL_WORKSPACE_XY_BOUNDS.
        """

        bounds = np.asarray(
            OFFICIAL_WORKSPACE_XY_BOUNDS,
            dtype=np.float64,
        ).reshape(2, 2).copy()

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise RuntimeError(
                "Official workspace XY bounds are invalid."
            )

        return bounds


    def _get_perception_workspace_limits(self):
        """Return the configured 3D MuJoCo workspace as [[x],[y],[z]]."""

        xy_bounds = self._get_perception_workspace_world_bounds()
        z_bounds = np.asarray(
            OFFICIAL_WORKSPACE_Z_LIMITS,
            dtype=np.float64,
        ).reshape(2)

        return np.array(
            [
                xy_bounds[0],
                xy_bounds[1],
                z_bounds,
            ],
            dtype=np.float64,
        )

    def _sample_workspace_drop_xy(self):
        """Sample XY inside the compact MuJoCo clutter rectangle."""

        bounds = np.asarray(
            self.object_drop_world_bounds,
            dtype=np.float64,
        )

        return np.array(
            [
                self.rng.uniform(
                    bounds[0, 0],
                    bounds[0, 1],
                ),
                self.rng.uniform(
                    bounds[1, 0],
                    bounds[1, 1],
                ),
            ],
            dtype=np.float64,
        )


    def _build_initial_clutter_by_center_drop(self):
        """Build clutter using PyBullet-style workspace-random dropping.

        For every attempt:
            1. park all selected objects away from the table;
            2. independently sample each object's XY inside the compact
               MuJoCo drop rectangle;
            3. drop one object;
            4. wait for it to settle;
            5. continue with the next object;
            6. reject and regenerate the whole scene if any object leaves
               the tabletop or purple perception workspace, or if any object
               failed to settle.

        The method name is retained for compatibility with existing callers.
        """

        table_top_z = float(
            self.table_offset[2]
            + self.table_full_size[2] / 2.0
        )

        self.initial_clutter_attempts = []

        for attempt_index in range(
            1,
            int(self.object_clutter_max_retries) + 1,
        ):
            self._park_all_clutter_objects()
            drop_results = []

            for obj in self.objects:
                drop_xy = self._sample_workspace_drop_xy()

                drop_position = np.array(
                    [
                        drop_xy[0],
                        drop_xy[1],
                        table_top_z
                        + self.object_drop_height_above_table,
                    ],
                    dtype=np.float64,
                )

                quaternion_wxyz = (
                    self._random_drop_quaternion_wxyz()
                )

                self._set_free_object_pose(
                    obj=obj,
                    position=drop_position,
                    quaternion_wxyz=quaternion_wxyz,
                )
                self.sim.forward()

                body_id = self.object_body_ids[obj.name]
                settle_result = (
                    self._wait_for_dropped_object_to_settle(
                        body_id=body_id,
                    )
                )

                drop_results.append(
                    {
                        "object_name": obj.name,
                        "drop_position": (
                            drop_position.copy()
                        ),
                        "drop_quaternion_wxyz": (
                            quaternion_wxyz.copy()
                        ),
                        **settle_result,
                    }
                )

            self.sim.forward()
            validity = self._validate_clutter_scene(
                drop_results=drop_results,
            )

            self.initial_clutter_attempts.append(
                {
                    "attempt": attempt_index,
                    "valid": bool(
                        validity["valid"]
                    ),
                    "invalid_objects": (
                        validity["invalid_objects"]
                    ),
                    "drop_results": drop_results,
                }
            )

            if validity["valid"]:
                self.initial_drop_results = (
                    drop_results
                )
                self.initial_clutter_validation = (
                    validity
                )
                self.initial_clutter_attempt_count = (
                    attempt_index
                )

                if attempt_index > 1:
                    print(
                        "Initial GSO clutter accepted after "
                        f"{attempt_index} attempts"
                    )

                return

            invalid_summary = ", ".join(
                (
                    f"{entry['object_name']}@"
                    f"{np.round(entry['position'], 4).tolist()}"
                    f"[table={entry['inside_table_xy']},"
                    f"purple={entry['inside_perception_xy']},"
                    f"settled={entry['settled']}]"
                )
                for entry in validity[
                    "invalid_objects"
                ]
            )

            print(
                "Initial GSO clutter attempt "
                f"{attempt_index} rejected: "
                f"{invalid_summary}"
            )

        raise RuntimeError(
            "Failed to generate a valid GSO clutter scene "
            f"after {self.object_clutter_max_retries} attempts."
        )


    def get_gripper_width(self):
        """Return the approximate opening width of the Panda gripper."""

        joint_positions = np.asarray(
            self.robots[0].get_gripper_joint_positions(),
            dtype=np.float64,
        )

        return float(np.sum(np.abs(joint_positions)))

    def command_gripper(
        self,
        command,
        steps=30,
    ):
        """Apply a gripper command while holding the arm pose.

        command:
            -1.0 opens the Panda gripper
            +1.0 closes the Panda gripper
        """

        action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )
        action[-1] = np.clip(
            command,
            -1.0,
            1.0,
        )

        for _ in range(steps):
            self.step(action)

        return {
            "joint_positions": np.asarray(
                self.robots[0].get_gripper_joint_positions()
            ).copy(),
            "width": self.get_gripper_width(),
        }

    def open_gripper(self, steps=30):
        """Open the Panda gripper."""

        return self.command_gripper(
            command=-1.0,
            steps=steps,
        )

    def close_gripper(self, steps=30):
        """Close the Panda gripper."""

        return self.command_gripper(
            command=1.0,
            steps=steps,
        )


    def has_gripper_table_contact(self):
        """Return True when any Panda gripper geom contacts the table."""

        table_body_id = self.sim.model.body_name2id("table")

        gripper_body_ids = set()

        for body_id in range(self.sim.model.nbody):
            body_name = self.sim.model.body_id2name(body_id)

            if (
                body_name
                and body_name.startswith("gripper0_right")
            ):
                gripper_body_ids.add(body_id)

        for contact_index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[contact_index]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            body1 = int(
                self.sim.model.geom_bodyid[geom1]
            )
            body2 = int(
                self.sim.model.geom_bodyid[geom2]
            )

            contact_is_gripper_table = (
                (
                    body1 in gripper_body_ids
                    and body2 == table_body_id
                )
                or
                (
                    body2 in gripper_body_ids
                    and body1 == table_body_id
                )
            )

            if contact_is_gripper_table:
                return True

        return False

    def get_eef_force_torque_wrench(self):
        """Read Panda end-effector force / torque sensors.

        The stopping metric is computed as sum(abs(wrench)) over the built-in
        force_ee and torque_ee MuJoCo sensors. The resulting scalar mixes force
        and torque units and is used only as a control threshold metric.
        """

        def _sensor_values_by_suffix(suffix):
            matches = []
            for sensor_id in range(self.sim.model.nsensor):
                sensor_name = self.sim.model.sensor_id2name(sensor_id)
                if sensor_name and sensor_name.endswith(suffix):
                    matches.append((sensor_id, sensor_name))

            if len(matches) != 1:
                raise RuntimeError(
                    "Expected exactly one MuJoCo sensor ending with "
                    f"{suffix!r}, found {matches}."
                )

            sensor_id, sensor_name = matches[0]
            address = int(self.sim.model.sensor_adr[sensor_id])
            dimension = int(self.sim.model.sensor_dim[sensor_id])
            values = np.asarray(
                self.sim.data.sensordata[address:address + dimension],
                dtype=np.float64,
            ).copy()
            return sensor_name, values

        force_name, force = _sensor_values_by_suffix("force_ee")
        torque_name, torque = _sensor_values_by_suffix("torque_ee")

        if force.shape != (3,) or torque.shape != (3,):
            raise RuntimeError(
                "Expected 3-axis force / torque sensors, got "
                f"force={force.shape}, torque={torque.shape}."
            )

        wrench = np.concatenate([force, torque])
        return {
            "force_sensor_name": force_name,
            "torque_sensor_name": torque_name,
            "force": force,
            "torque": torque,
            "wrench": wrench,
            "source_style_metric": float(np.sum(np.abs(wrench))),
        }

    def get_eef_pose(self, arm="right"):
        """Return the Panda EEF site pose in MuJoCo world coordinates."""

        site_ids = self.robots[0].eef_site_id

        if isinstance(site_ids, dict):
            if arm not in site_ids:
                raise KeyError(f"Unknown arm: {arm}")
            site_id = site_ids[arm]
        else:
            site_id = site_ids

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = (
            self.sim.data.site_xmat[site_id]
            .reshape(3, 3)
            .copy()
        )
        pose[:3, 3] = self.sim.data.site_xpos[site_id].copy()

        return pose

    def get_eef_velocity(self, arm="right"):
        """Return EEF linear and angular velocity in world coordinates."""

        site_ids = self.robots[0].eef_site_id

        if isinstance(site_ids, dict):
            if arm not in site_ids:
                raise KeyError(f"Unknown arm: {arm}")
            site_id = site_ids[arm]
        else:
            site_id = site_ids

        site_name = self.sim.model.site_id2name(site_id)

        linear_velocity = np.asarray(
            self.sim.data.get_site_xvelp(site_name),
            dtype=np.float64,
        ).copy()
        angular_velocity = np.asarray(
            self.sim.data.get_site_xvelr(site_name),
            dtype=np.float64,
        ).copy()

        return linear_velocity, angular_velocity

    def get_arm_joint_positions(self):
        """Return the current Panda arm joint positions (7 DoF)."""

        joint_indexes = np.asarray(
            self.robots[0]._ref_joint_pos_indexes,
            dtype=np.int64,
        )

        return self.sim.data.qpos[joint_indexes].copy()

    def _native_mujoco_model_and_data(self):
        """Return native mujoco.MjModel / MjData behind robosuite bindings."""

        model = getattr(
            self.sim.model,
            "_model",
            self.sim.model,
        )
        data = getattr(
            self.sim.data,
            "_data",
            self.sim.data,
        )

        return model, data

    def _require_joint_position_controller(
        self,
        arm="right",
    ):
        """Return the active arm controller and verify JOINT_POSITION support."""

        controller = self.robots[0].part_controllers[arm]

        controller_name = str(
            getattr(
                controller,
                "name",
                controller.__class__.__name__,
            )
        ).upper()

        if (
            "JOINT_POSITION" not in controller_name
            and "JOINTPOSITION" not in controller_name
        ):
            raise RuntimeError(
                "IK / q_ref execution requires the Panda right arm to use "
                "robosuite JOINT_POSITION. Current controller is "
                f"{controller.__class__.__name__!r}. "
                "Instantiate the environment with "
                "load_thinkgrasp_joint_position_controller_config()."
            )

        if not hasattr(controller, "set_goal"):
            raise RuntimeError(
                "Active JOINT_POSITION controller has no set_goal() method."
            )

        return controller

    def _solve_ik_single_seed(
        self,
        target_pose,
        arm="right",
        initial_joint_positions=None,
        max_iterations=100,
        residual_threshold=1e-5,
        damping=1e-4,
        step_scale=0.8,
    ):
        """Solve Panda 7-DoF IK with MuJoCo site Jacobians.

        The solver temporarily writes arm qpos for FK / Jacobian evaluation,
        then restores the exact simulator state before returning.

        Args:
            target_pose:
                4x4 world-frame target pose for Panda eef_site.
            initial_joint_positions:
                Optional 7-vector seed. Current arm q is used by default.

        Returns:
            dict with convergence diagnostics and the best joint solution.

        Execution policy:
            - residual_threshold is used only for early convergence;
            - reaching max_iterations without satisfying the threshold does
              not automatically forbid execution;
            - downstream JOINT_POSITION execution may still try the best
              finite joint solution returned here.
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float64,
        )

        if target_pose.shape != (4, 4):
            raise ValueError(
                "target_pose must have shape (4, 4), "
                f"got {target_pose.shape}"
            )

        robot = self.robots[0]

        qpos_indexes = np.asarray(
            robot._ref_joint_pos_indexes,
            dtype=np.int64,
        )
        dof_indexes = np.asarray(
            robot._ref_joint_vel_indexes,
            dtype=np.int64,
        )

        if len(qpos_indexes) != 7 or len(dof_indexes) != 7:
            raise RuntimeError(
                "Expected Panda 7-DoF arm indexes, got "
                f"qpos={len(qpos_indexes)}, dof={len(dof_indexes)}"
            )

        if initial_joint_positions is None:
            q = self.get_arm_joint_positions().astype(
                np.float64
            )
        else:
            q = np.asarray(
                initial_joint_positions,
                dtype=np.float64,
            ).reshape(7).copy()

        site_ids = robot.eef_site_id

        if isinstance(site_ids, dict):
            if arm not in site_ids:
                raise KeyError(f"Unknown arm: {arm}")
            site_id = int(site_ids[arm])
        else:
            site_id = int(site_ids)

        model, data = self._native_mujoco_model_and_data()

        saved_qpos = self.sim.data.qpos.copy()
        saved_qvel = self.sim.data.qvel.copy()
        saved_ctrl = self.sim.data.ctrl.copy()

        joint_ids = []

        for qpos_index in qpos_indexes:
            matching_joint_ids = np.where(
                np.asarray(
                    self.sim.model.jnt_qposadr,
                    dtype=np.int64,
                )
                == int(qpos_index)
            )[0]

            if len(matching_joint_ids) != 1:
                raise RuntimeError(
                    "Could not uniquely map Panda qpos index "
                    f"{qpos_index} to a MuJoCo joint."
                )

            joint_ids.append(
                int(matching_joint_ids[0])
            )

        lower_limits = np.full(
            7,
            -np.inf,
            dtype=np.float64,
        )
        upper_limits = np.full(
            7,
            np.inf,
            dtype=np.float64,
        )

        for index, joint_id in enumerate(joint_ids):
            limited = bool(
                self.sim.model.jnt_limited[
                    joint_id
                ]
            )

            if limited:
                joint_range = np.asarray(
                    self.sim.model.jnt_range[
                        joint_id
                    ],
                    dtype=np.float64,
                )

                lower_limits[index] = joint_range[0]
                upper_limits[index] = joint_range[1]

        target_position = target_pose[:3, 3]
        target_rotation = target_pose[:3, :3]

        success = False
        iterations = 0
        position_error_norm = np.inf
        orientation_error_norm = np.inf

        try:
            for iteration in range(
                int(max_iterations) + 1
            ):
                self.sim.data.qpos[
                    qpos_indexes
                ] = q

                self.sim.data.qvel[
                    dof_indexes
                ] = 0.0

                mujoco.mj_forward(
                    model,
                    data,
                )

                current_position = np.asarray(
                    self.sim.data.site_xpos[
                        site_id
                    ],
                    dtype=np.float64,
                ).copy()

                current_rotation = np.asarray(
                    self.sim.data.site_xmat[
                        site_id
                    ],
                    dtype=np.float64,
                ).reshape(3, 3).copy()

                position_error = (
                    target_position
                    - current_position
                )

                orientation_error = (
                    SciPyRotation.from_matrix(
                        target_rotation
                        @ current_rotation.T
                    ).as_rotvec()
                )

                position_error_norm = float(
                    np.linalg.norm(
                        position_error
                    )
                )
                orientation_error_norm = float(
                    np.linalg.norm(
                        orientation_error
                    )
                )

                combined_error = np.concatenate(
                    [
                        position_error,
                        orientation_error,
                    ]
                )

                residual = float(
                    np.linalg.norm(
                        combined_error
                    )
                )

                iterations = iteration

                if residual <= float(
                    residual_threshold
                ):
                    success = True
                    break

                if iteration >= int(
                    max_iterations
                ):
                    break

                jac_position = np.zeros(
                    (3, model.nv),
                    dtype=np.float64,
                )
                jac_rotation = np.zeros(
                    (3, model.nv),
                    dtype=np.float64,
                )

                mujoco.mj_jacSite(
                    model,
                    data,
                    jac_position,
                    jac_rotation,
                    site_id,
                )

                jacobian = np.vstack(
                    [
                        jac_position[
                            :,
                            dof_indexes,
                        ],
                        jac_rotation[
                            :,
                            dof_indexes,
                        ],
                    ]
                )

                damping_value = float(
                    damping
                )

                regularized = (
                    jacobian @ jacobian.T
                    + (
                        damping_value**2
                        * np.eye(
                            6,
                            dtype=np.float64,
                        )
                    )
                )

                delta_q = (
                    jacobian.T
                    @ np.linalg.solve(
                        regularized,
                        combined_error,
                    )
                )

                q += (
                    float(step_scale)
                    * delta_q
                )

                q = np.minimum(
                    np.maximum(
                        q,
                        lower_limits,
                    ),
                    upper_limits,
                )

        finally:
            self.sim.data.qpos[:] = saved_qpos
            self.sim.data.qvel[:] = saved_qvel
            self.sim.data.ctrl[:] = saved_ctrl

            mujoco.mj_forward(
                model,
                data,
            )

        usable_solution = bool(
            np.all(
                np.isfinite(q)
            )
        )

        return {
            # Keep strict residual convergence visible for diagnostics.
            "success": bool(success),
            "converged": bool(success),
            # A finite best-effort solution is executable, matching the
            # original PyBullet flow more closely.
            "usable_solution": usable_solution,
            "iterations": int(iterations),
            "joint_positions": q.copy(),
            "position_error_norm": float(
                position_error_norm
            ),
            "orientation_error_norm": float(
                orientation_error_norm
            ),
            "orientation_error_deg": float(
                np.rad2deg(
                    orientation_error_norm
                )
            ),
            "residual_threshold": float(
                residual_threshold
            ),
        }

    def solve_ik(
        self,
        target_pose,
        arm="right",
        initial_joint_positions=None,
        max_iterations=100,
        residual_threshold=1e-5,
        damping=1e-4,
        step_scale=0.8,
    ):
        """Solve Panda IK from several virtual seeds and return the best one.

        The real robot does NOT move while these seeds are evaluated.
        Seed 0 is exactly the original current / caller-supplied joint state.

        Multi-start changes only the initial guesses used by IK.
        The original single-seed Jacobian solver, iteration count,
        residual threshold, damping, step scale, and JOINT_POSITION
        execution remain unchanged.
        """

        if initial_joint_positions is None:
            base_seed = (
                self.get_arm_joint_positions()
                .astype(np.float64)
            )
        else:
            base_seed = np.asarray(
                initial_joint_positions,
                dtype=np.float64,
            ).reshape(7).copy()

        robot = self.robots[0]

        qpos_indexes = np.asarray(
            robot._ref_joint_pos_indexes,
            dtype=np.int64,
        )

        joint_ids = []

        for qpos_index in qpos_indexes:
            matching_joint_ids = np.where(
                np.asarray(
                    self.sim.model.jnt_qposadr,
                    dtype=np.int64,
                )
                == int(qpos_index)
            )[0]

            if len(matching_joint_ids) != 1:
                raise RuntimeError(
                    "Could not uniquely map Panda qpos index "
                    f"{qpos_index} to a MuJoCo joint."
                )

            joint_ids.append(
                int(matching_joint_ids[0])
            )

        lower_limits = np.full(
            7,
            -np.inf,
            dtype=np.float64,
        )

        upper_limits = np.full(
            7,
            np.inf,
            dtype=np.float64,
        )

        for index, joint_id in enumerate(
            joint_ids
        ):
            if bool(
                self.sim.model.jnt_limited[
                    joint_id
                ]
            ):
                joint_range = np.asarray(
                    self.sim.model.jnt_range[
                        joint_id
                    ],
                    dtype=np.float64,
                )

                lower_limits[index] = (
                    joint_range[0]
                )
                upper_limits[index] = (
                    joint_range[1]
                )

        # Deterministic virtual IK seeds.
        # These are calculation-only initial guesses.
        # The physical Panda never moves to these poses.
        seed_offsets = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                [0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                [0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.35, 0.0, 0.0, 0.0, 0.0],

                [0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.35, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        seeds = []

        for offset in seed_offsets:
            seed = np.minimum(
                np.maximum(
                    base_seed + offset,
                    lower_limits,
                ),
                upper_limits,
            )

            # Avoid duplicate seeds after joint-limit clipping.
            duplicate = any(
                np.allclose(
                    seed,
                    existing_seed,
                    atol=1e-10,
                    rtol=0.0,
                )
                for existing_seed in seeds
            )

            if not duplicate:
                seeds.append(seed)

        candidate_results = []

        for seed_index, seed in enumerate(
            seeds
        ):
            result = (
                self._solve_ik_single_seed(
                    target_pose=target_pose,
                    arm=arm,
                    initial_joint_positions=seed,
                    max_iterations=max_iterations,
                    residual_threshold=(
                        residual_threshold
                    ),
                    damping=damping,
                    step_scale=step_scale,
                )
            )

            # Same position[m] + orientation[rad] norm
            # already used by the existing solver.
            # Here it is used only to RANK seeds.
            candidate_residual = float(
                np.hypot(
                    result[
                        "position_error_norm"
                    ],
                    result[
                        "orientation_error_norm"
                    ],
                )
            )

            candidate_results.append(
                {
                    "seed_index": int(
                        seed_index
                    ),
                    "seed_joint_positions": (
                        seed.copy()
                    ),
                    "result": result,
                    "residual": (
                        candidate_residual
                    ),
                }
            )

        usable_candidates = [
            candidate
            for candidate in candidate_results
            if (
                candidate["result"][
                    "usable_solution"
                ]
                and np.isfinite(
                    candidate["residual"]
                )
            )
        ]

        # Continuity-first IK preference:
        #
        # Seed 0 is always the current or caller-supplied Panda joint
        # configuration. During straight Cartesian motion this is therefore
        # the solution reached for the preceding waypoint.
        #
        # If seed 0 already gives a practically acceptable Cartesian pose,
        # keep that joint branch. Other seeds are used only as fallback.
        continuity_candidate = candidate_results[0]
        continuity_result = continuity_candidate["result"]

        continuity_candidate_acceptable = bool(
            continuity_result["converged"]
            or (
                continuity_result["usable_solution"]
                and continuity_result["position_error_norm"]
                <= IK_BEST_EFFORT_MAX_POSITION_ERROR_M
                and continuity_result["orientation_error_deg"]
                <= IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG
            )
        )

        if continuity_candidate_acceptable:
            best_candidate = continuity_candidate
            ik_selection_mode = "continuity_seed0"
        elif usable_candidates:
            best_candidate = min(
                usable_candidates,
                key=lambda candidate: (
                    candidate["residual"]
                ),
            )
            ik_selection_mode = "multi_start_fallback"
        else:
            best_candidate = min(
                candidate_results,
                key=lambda candidate: (
                    candidate["residual"]
                    if np.isfinite(
                        candidate["residual"]
                    )
                    else np.inf
                ),
            )
            ik_selection_mode = "multi_start_fallback"

        best_result = dict(
            best_candidate["result"]
        )

        # Preserve multi-start diagnostics.
        best_result.update(
            {
                "multi_start": True,
                "multi_start_seed_count": (
                    len(seeds)
                ),
                "selected_seed_index": int(
                    best_candidate[
                        "seed_index"
                    ]
                ),
                "selected_seed_joint_positions": (
                    best_candidate[
                        "seed_joint_positions"
                    ].copy()
                ),
                "selected_residual": float(
                    best_candidate[
                        "residual"
                    ]
                ),
                "ik_selection_mode": (
                    ik_selection_mode
                ),
                "multi_start_candidates": [
                    {
                        "seed_index": int(
                            candidate[
                                "seed_index"
                            ]
                        ),
                        "converged": bool(
                            candidate["result"][
                                "converged"
                            ]
                        ),
                        "usable_solution": bool(
                            candidate["result"][
                                "usable_solution"
                            ]
                        ),
                        "position_error_norm": float(
                            candidate["result"][
                                "position_error_norm"
                            ]
                        ),
                        "orientation_error_deg": float(
                            candidate["result"][
                                "orientation_error_deg"
                            ]
                        ),
                        "residual": float(
                            candidate[
                                "residual"
                            ]
                        ),
                    }
                    for candidate
                    in candidate_results
                ],
            }
        )

        return best_result


    def move_joints_qref(
        self,
        target_joint_positions,
        arm="right",
        qref_speed_rad_per_sec=IK_QREF_SPEED_RAD_PER_SEC,
        joint_tolerance=0.02,
        max_physics_steps=6000,
        gripper_command=0.0,
        stop_on_table_contact=False,
        force_stop_enabled=False,
        force_stop_threshold=PYBULLET_STYLE_GRASP_FORCE_THRESHOLD,
        force_stop_consecutive_steps=GRASP_FORCE_STOP_CONSECUTIVE_STEPS,
        frame_callback=None,
        frame_capture_interval=10,
    ):
        """Move Panda joints with the validated independent-q_ref controller.

        Unlike the first failed prototype, q_ref advances independently from
        the measured q. If the physical joints lag, the tracking error grows
        naturally and the JOINT_POSITION controller supplies more torque.

        frame_callback, when supplied, is called periodically as:
            frame_callback(self)

        The environment itself stays video-agnostic; run_closed_loop.py may
        later use this hook to record every motion phase.
        """

        target_joint_positions = np.asarray(
            target_joint_positions,
            dtype=np.float64,
        ).reshape(7)

        controller = (
            self._require_joint_position_controller(
                arm=arm
            )
        )

        controller.kp = np.full(
            7,
            IK_JOINT_KP,
            dtype=np.float64,
        )
        controller.kd = (
            2.0
            * np.sqrt(
                controller.kp
            )
            * IK_JOINT_DAMPING_RATIO
        )
        controller.goal_qpos = None

        robot = self.robots[0]

        qpos_indexes = np.asarray(
            robot._ref_joint_pos_indexes,
            dtype=np.int64,
        )

        arm_actuator_ids = np.asarray(
            [
                self.sim.model.actuator_name2id(
                    f"robot0_torq_j{i}"
                )
                for i in range(1, 8)
            ],
            dtype=np.int64,
        )

        actuator_ctrlrange = np.asarray(
            self.sim.model.actuator_ctrlrange[
                arm_actuator_ids
            ],
            dtype=np.float64,
        )

        model, data = (
            self._native_mujoco_model_and_data()
        )

        physics_dt = float(
            self.sim.model.opt.timestep
        )

        qref_step = (
            float(qref_speed_rad_per_sec)
            * physics_dt
        )

        if qref_step <= 0.0:
            raise ValueError(
                "qref_speed_rad_per_sec must be positive"
            )

        q_ref = self.get_arm_joint_positions().astype(
            np.float64
        )

        start_time = float(
            self.sim.data.time
        )
        max_tracking_error_seen = 0.0
        max_raw_torque_seen = 0.0

        # Diagnostic only: record contact pairs seen while q_ref motion runs.
        # This does not stop, slow, or otherwise alter robot execution.
        # Keep only contacts likely to explain clutter jamming:
        #   - any contact involving a Panda gripper body;
        #   - contacts between two tracked clutter objects.
        diagnostic_contact_pairs = set()
        max_relevant_contact_count = 0

        force_stop_consecutive_steps = max(1, int(force_stop_consecutive_steps))
        force_stop_counter = 0
        max_eef_force_metric_seen = 0.0
        last_eef_wrench = None
        clutter_body_ids = set(
            int(body_id)
            for body_id in self.object_body_ids.values()
        )

        frame_capture_interval = max(
            1,
            int(frame_capture_interval),
        )

        for physics_step in range(
            1,
            int(max_physics_steps) + 1,
        ):
            controller.update(force=True)

            current_q = np.asarray(
                self.sim.data.qpos[
                    qpos_indexes
                ],
                dtype=np.float64,
            ).copy()

            joint_error = (
                target_joint_positions
                - current_q
            )

            max_abs_joint_error = float(
                np.max(
                    np.abs(
                        joint_error
                    )
                )
            )

            if (
                max_abs_joint_error
                < float(joint_tolerance)
            ):
                final_pose = self.get_eef_pose(
                    arm=arm
                )

                return {
                    "success": True,
                    "failure_reason": None,
                    "physics_steps": physics_step - 1,
                    "simulated_seconds": (
                        float(self.sim.data.time)
                        - start_time
                    ),
                    "target_joint_positions": (
                        target_joint_positions.copy()
                    ),
                    "final_joint_positions": (
                        current_q
                    ),
                    "max_abs_joint_error": (
                        max_abs_joint_error
                    ),
                    "max_tracking_error_seen": (
                        max_tracking_error_seen
                    ),
                    "max_raw_torque_seen": (
                        max_raw_torque_seen
                    ),
                    "stopped_by_force": False,
                    "max_eef_force_metric_seen": (
                        float(max_eef_force_metric_seen)
                        if force_stop_enabled
                        else None
                    ),
                    "diagnostic_contact_pairs": sorted(
                        diagnostic_contact_pairs
                    ),
                    "max_relevant_contact_count": int(
                        max_relevant_contact_count
                    ),
                    "final_pose": final_pose,
                }

            ref_error = (
                target_joint_positions
                - q_ref
            )

            ref_error_norm = float(
                np.linalg.norm(
                    ref_error
                )
            )

            if ref_error_norm > 1e-12:
                q_ref += (
                    ref_error
                    / ref_error_norm
                    * min(
                        qref_step,
                        ref_error_norm,
                    )
                )

            tracking_error = (
                q_ref - current_q
            )

            max_tracking_error_seen = max(
                max_tracking_error_seen,
                float(
                    np.max(
                        np.abs(
                            tracking_error
                        )
                    )
                ),
            )

            controller.set_goal(
                np.zeros(
                    controller.control_dim,
                    dtype=np.float64,
                ),
                set_qpos=q_ref,
            )

            torques = np.asarray(
                controller.run_controller(),
                dtype=np.float64,
            )

            max_raw_torque_seen = max(
                max_raw_torque_seen,
                float(
                    np.max(
                        np.abs(
                            torques
                        )
                    )
                ),
            )

            torques = np.clip(
                torques,
                actuator_ctrlrange[:, 0],
                actuator_ctrlrange[:, 1],
            )

            self.sim.data.ctrl[
                arm_actuator_ids
            ] = torques

            # Keep the requested gripper command active while raw MuJoCo
            # physics advances. The Panda finger actuators remain controlled
            # by robosuite only through normal env.step(), so q_ref motion
            # intentionally leaves their current actuator commands untouched.
            mujoco.mj_step(
                model,
                data,
            )

            # Diagnostic only. Observe contacts after this physics step,
            # without changing any control command or failure condition.
            relevant_contacts_this_step = 0

            for contact_index in range(self.sim.data.ncon):
                contact = self.sim.data.contact[contact_index]

                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                body1 = int(self.sim.model.geom_bodyid[geom1])
                body2 = int(self.sim.model.geom_bodyid[geom2])

                name1 = self.sim.model.body_id2name(body1) or f"body_{body1}"
                name2 = self.sim.model.body_id2name(body2) or f"body_{body2}"

                body1_is_gripper = str(name1).startswith(
                    "gripper0_right"
                )
                body2_is_gripper = str(name2).startswith(
                    "gripper0_right"
                )
                both_are_clutter = (
                    body1 in clutter_body_ids
                    and body2 in clutter_body_ids
                )

                if not (
                    body1_is_gripper
                    or body2_is_gripper
                    or both_are_clutter
                ):
                    continue

                relevant_contacts_this_step += 1
                pair = tuple(sorted((str(name1), str(name2))))
                diagnostic_contact_pairs.add(
                    f"{pair[0]} <-> {pair[1]}"
                )

            max_relevant_contact_count = max(
                max_relevant_contact_count,
                relevant_contacts_this_step,
            )

            if (
                stop_on_table_contact
                and self.has_gripper_table_contact()
            ):
                return {
                    "success": False,
                    "failure_reason": "table_contact",
                    "physics_steps": physics_step,
                    "simulated_seconds": (
                        float(self.sim.data.time)
                        - start_time
                    ),
                    "target_joint_positions": (
                        target_joint_positions.copy()
                    ),
                    "final_joint_positions": (
                        self.get_arm_joint_positions()
                    ),
                    "max_abs_joint_error": float(
                        np.max(
                            np.abs(
                                target_joint_positions
                                - self.get_arm_joint_positions()
                            )
                        )
                    ),
                    "max_tracking_error_seen": (
                        max_tracking_error_seen
                    ),
                    "max_raw_torque_seen": (
                        max_raw_torque_seen
                    ),
                    "stopped_by_force": False,
                    "max_eef_force_metric_seen": (
                        float(max_eef_force_metric_seen)
                        if force_stop_enabled
                        else None
                    ),
                    "diagnostic_contact_pairs": sorted(
                        diagnostic_contact_pairs
                    ),
                    "max_relevant_contact_count": int(
                        max_relevant_contact_count
                    ),
                    "final_pose": self.get_eef_pose(
                        arm=arm
                    ),
                }

            if force_stop_enabled:
                wrench_result = self.get_eef_force_torque_wrench()
                force_metric = float(wrench_result["source_style_metric"])
                last_eef_wrench = wrench_result["wrench"].copy()
                max_eef_force_metric_seen = max(
                    max_eef_force_metric_seen, force_metric
                )

                if force_metric > float(force_stop_threshold):
                    force_stop_counter += 1
                else:
                    force_stop_counter = 0

                if force_stop_counter >= force_stop_consecutive_steps:
                    return {
                        "success": True,
                        "failure_reason": None,
                        "stopped_by_force": True,
                        "force_stop_threshold": float(force_stop_threshold),
                        "force_stop_consecutive_steps": int(force_stop_consecutive_steps),
                        "force_stop_metric": force_metric,
                        "force_stop_wrench": last_eef_wrench.copy(),
                        "max_eef_force_metric_seen": float(max_eef_force_metric_seen),
                        "physics_steps": physics_step,
                        "simulated_seconds": float(self.sim.data.time) - start_time,
                        "target_joint_positions": target_joint_positions.copy(),
                        "final_joint_positions": self.get_arm_joint_positions(),
                        "max_abs_joint_error": float(
                            np.max(np.abs(target_joint_positions - self.get_arm_joint_positions()))
                        ),
                        "max_tracking_error_seen": max_tracking_error_seen,
                        "max_raw_torque_seen": max_raw_torque_seen,
                        "diagnostic_contact_pairs": sorted(diagnostic_contact_pairs),
                        "max_relevant_contact_count": int(max_relevant_contact_count),
                        "final_pose": self.get_eef_pose(arm=arm),
                    }

            if (
                frame_callback is not None
                and physics_step
                % frame_capture_interval
                == 0
            ):
                frame_callback(self)

        return {
            "success": False,
            "failure_reason": "timeout",
            "physics_steps": int(
                max_physics_steps
            ),
            "simulated_seconds": (
                float(self.sim.data.time)
                - start_time
            ),
            "target_joint_positions": (
                target_joint_positions.copy()
            ),
            "final_joint_positions": (
                self.get_arm_joint_positions()
            ),
            "max_abs_joint_error": float(
                np.max(
                    np.abs(
                        target_joint_positions
                        - self.get_arm_joint_positions()
                    )
                )
            ),
            "max_tracking_error_seen": (
                max_tracking_error_seen
            ),
            "max_raw_torque_seen": (
                max_raw_torque_seen
            ),
            "stopped_by_force": False,
            "max_eef_force_metric_seen": (
                float(max_eef_force_metric_seen)
                if force_stop_enabled
                else None
            ),
            "diagnostic_contact_pairs": sorted(
                diagnostic_contact_pairs
            ),
            "max_relevant_contact_count": int(
                max_relevant_contact_count
            ),
            "final_pose": self.get_eef_pose(
                arm=arm
            ),
        }

    def move_pose_ik(
        self,
        target_pose,
        arm="right",
        initial_joint_positions=None,
        joint_tolerance=0.02,
        qref_speed_rad_per_sec=IK_QREF_SPEED_RAD_PER_SEC,
        max_physics_steps=6000,
        stop_on_table_contact=False,
        force_stop_enabled=False,
        force_stop_threshold=PYBULLET_STYLE_GRASP_FORCE_THRESHOLD,
        force_stop_consecutive_steps=GRASP_FORCE_STOP_CONSECUTIVE_STEPS,
        frame_callback=None,
        frame_capture_interval=10,
    ):
        """Solve IK for one EEF target pose and execute it with q_ref."""

        if initial_joint_positions is None:
            initial_joint_positions = (
                self.get_arm_joint_positions()
            )

        ik_result = self.solve_ik(
            target_pose=target_pose,
            arm=arm,
            initial_joint_positions=(
                initial_joint_positions
            ),
        )

        # Keep the original strict residual threshold as a convergence
        # diagnostic, but do not require every practically usable solution
        # to reach 1e-5.
        #
        # Multi-start can still return the "best" candidate even when every
        # seed is poor. Therefore a non-converged solution is executable only
        # when its actual Cartesian position AND orientation errors are
        # already within explicit practical tolerances.
        strict_converged = bool(
            ik_result["converged"]
        )

        acceptable_best_effort = bool(
            ik_result["usable_solution"]
            and ik_result["position_error_norm"]
            <= IK_BEST_EFFORT_MAX_POSITION_ERROR_M
            and ik_result["orientation_error_deg"]
            <= IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG
        )

        if strict_converged:
            ik_execution_decision = (
                "strict_converged"
            )
        elif acceptable_best_effort:
            ik_execution_decision = (
                "accepted_best_effort"
            )
            print(
                "IK BEST-EFFORT ACCEPTED: "
                f"seed={ik_result.get('selected_seed_index')}, "
                f"position_error="
                f"{1000.0 * ik_result['position_error_norm']:.3f} mm, "
                f"orientation_error="
                f"{ik_result['orientation_error_deg']:.3f} deg"
            )
        else:
            ik_execution_decision = (
                "rejected_bad_ik"
            )

            print(
                "IK QUALITY GATE REJECTED: "
                f"seed={ik_result.get('selected_seed_index')}, "
                f"position_error="
                f"{1000.0 * ik_result['position_error_norm']:.3f} mm, "
                f"orientation_error="
                f"{ik_result['orientation_error_deg']:.3f} deg, "
                f"limits="
                f"{1000.0 * IK_BEST_EFFORT_MAX_POSITION_ERROR_M:.1f} mm/"
                f"{IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG:.1f} deg"
            )

            candidates = ik_result.get(
                "multi_start_candidates",
                [],
            )

            if candidates:
                print(
                    "IK multi-start rejected:",
                    f"{len(candidates)} candidates evaluated"
                )

            ik_result = dict(ik_result)
            ik_result[
                "execution_decision"
            ] = ik_execution_decision
            ik_result[
                "best_effort_position_limit_m"
            ] = float(
                IK_BEST_EFFORT_MAX_POSITION_ERROR_M
            )
            ik_result[
                "best_effort_orientation_limit_deg"
            ] = float(
                IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG
            )

            return {
                "success": False,
                "failure_reason": "ik_failed",
                "ik": ik_result,
                "motion": None,
                "target_pose": np.asarray(
                    target_pose,
                    dtype=np.float64,
                ).copy(),
            }

        ik_result = dict(ik_result)
        ik_result[
            "execution_decision"
        ] = ik_execution_decision
        ik_result[
            "best_effort_position_limit_m"
        ] = float(
            IK_BEST_EFFORT_MAX_POSITION_ERROR_M
        )
        ik_result[
            "best_effort_orientation_limit_deg"
        ] = float(
            IK_BEST_EFFORT_MAX_ORIENTATION_ERROR_DEG
        )

        motion_result = self.move_joints_qref(
            target_joint_positions=(
                ik_result["joint_positions"]
            ),
            arm=arm,
            qref_speed_rad_per_sec=(
                qref_speed_rad_per_sec
            ),
            joint_tolerance=joint_tolerance,
            max_physics_steps=max_physics_steps,
            stop_on_table_contact=(
                stop_on_table_contact
            ),
            force_stop_enabled=force_stop_enabled,
            force_stop_threshold=force_stop_threshold,
            force_stop_consecutive_steps=force_stop_consecutive_steps,
            frame_callback=frame_callback,
            frame_capture_interval=(
                frame_capture_interval
            ),
        )

        final_pose = self.get_eef_pose(
            arm=arm
        )
        target_pose_array = np.asarray(
            target_pose,
            dtype=np.float64,
        )

        position_error = (
            target_pose_array[:3, 3]
            - final_pose[:3, 3]
        )

        orientation_error = (
            SciPyRotation.from_matrix(
                target_pose_array[:3, :3]
                @ final_pose[:3, :3].T
            ).as_rotvec()
        )

        return {
            "success": bool(
                motion_result["success"]
            ),
            "failure_reason": (
                motion_result[
                    "failure_reason"
                ]
            ),
            "stopped_by_force": bool(motion_result.get("stopped_by_force", False)),
            "force_stop_metric": motion_result.get("force_stop_metric"),
            "force_stop_wrench": motion_result.get("force_stop_wrench"),
            "ik": ik_result,
            "ik_converged_before_execution": bool(
                ik_result["converged"]
            ),
            "ik_best_effort_executed": bool(
                not ik_result["converged"]
                and ik_result["usable_solution"]
            ),
            "motion": motion_result,
            "target_pose": (
                target_pose_array.copy()
            ),
            "final_pose": final_pose,
            "position_error": position_error,
            "position_error_norm": float(
                np.linalg.norm(
                    position_error
                )
            ),
            "orientation_error": (
                orientation_error
            ),
            "orientation_error_norm": float(
                np.linalg.norm(
                    orientation_error
                )
            ),
            "orientation_error_deg": float(
                np.rad2deg(
                    np.linalg.norm(
                        orientation_error
                    )
                )
            ),
        }

    def straight_move_ik(
        self,
        target_pose,
        arm="right",
        waypoint_spacing=IK_CARTESIAN_WAYPOINT_SPACING,
        joint_tolerance=0.012,
        qref_speed_rad_per_sec=IK_QREF_SPEED_RAD_PER_SEC,
        stop_on_table_contact=False,
        force_stop_enabled=False,
        force_stop_threshold=PYBULLET_STYLE_GRASP_FORCE_THRESHOLD,
        force_stop_consecutive_steps=GRASP_FORCE_STOP_CONSECUTIVE_STEPS,
        frame_callback=None,
        frame_capture_interval=10,
    ):
        """Move EEF to target through approximately 1 cm Cartesian waypoints.

        Position and orientation are both interpolated. Each waypoint is
        solved by IK and then executed by q_ref + JOINT_POSITION.
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float64,
        )

        if target_pose.shape != (4, 4):
            raise ValueError(
                "target_pose must have shape (4, 4), "
                f"got {target_pose.shape}"
            )

        start_pose = self.get_eef_pose(
            arm=arm
        )

        translation = (
            target_pose[:3, 3]
            - start_pose[:3, 3]
        )

        distance = float(
            np.linalg.norm(
                translation
            )
        )

        waypoint_spacing = max(
            float(waypoint_spacing),
            1e-6,
        )

        waypoint_count = max(
            1,
            int(
                np.ceil(
                    distance
                    / waypoint_spacing
                )
            ),
        )

        relative_rotvec = (
            SciPyRotation.from_matrix(
                target_pose[:3, :3]
                @ start_pose[:3, :3].T
            ).as_rotvec()
        )

        waypoint_results = []

        for waypoint_index in range(
            1,
            waypoint_count + 1,
        ):
            alpha = (
                waypoint_index
                / float(
                    waypoint_count
                )
            )

            waypoint_pose = np.eye(
                4,
                dtype=np.float64,
            )

            waypoint_pose[:3, 3] = (
                start_pose[:3, 3]
                + alpha * translation
            )

            waypoint_pose[:3, :3] = (
                SciPyRotation.from_rotvec(
                    alpha * relative_rotvec
                ).as_matrix()
                @ start_pose[:3, :3]
            )

            result = self.move_pose_ik(
                target_pose=waypoint_pose,
                arm=arm,
                joint_tolerance=(
                    joint_tolerance
                ),
                qref_speed_rad_per_sec=(
                    qref_speed_rad_per_sec
                ),
                stop_on_table_contact=(
                    stop_on_table_contact
                ),
                force_stop_enabled=force_stop_enabled,
                force_stop_threshold=force_stop_threshold,
                force_stop_consecutive_steps=force_stop_consecutive_steps,
                frame_callback=(
                    frame_callback
                ),
                frame_capture_interval=(
                    frame_capture_interval
                ),
            )

            waypoint_results.append(
                result
            )

            if result.get("stopped_by_force", False):
                return {
                    "success": True,
                    "failure_reason": None,
                    "stopped_by_force": True,
                    "force_stop_metric": result.get("force_stop_metric"),
                    "force_stop_wrench": result.get("force_stop_wrench"),
                    "stopped_waypoint": waypoint_index,
                    "failed_waypoint": None,
                    "completed_waypoints": waypoint_index - 1,
                    "waypoint_count": waypoint_count,
                    "waypoints": waypoint_results,
                    "target_pose": target_pose.copy(),
                }

            if not result["success"]:
                return {
                    "success": False,
                    "failure_reason": (
                        result[
                            "failure_reason"
                        ]
                    ),
                    "failed_waypoint": (
                        waypoint_index
                    ),
                    "completed_waypoints": (
                        waypoint_index - 1
                    ),
                    "waypoint_count": (
                        waypoint_count
                    ),
                    "waypoints": (
                        waypoint_results
                    ),
                    "target_pose": (
                        target_pose.copy()
                    ),
                }

        return {
            "success": True,
            "failure_reason": None,
            "stopped_by_force": False,
            "force_stop_metric": None,
            "force_stop_wrench": None,
            "failed_waypoint": None,
            "completed_waypoints": (
                waypoint_count
            ),
            "waypoint_count": (
                waypoint_count
            ),
            "waypoints": (
                waypoint_results
            ),
            "target_pose": (
                target_pose.copy()
            ),
        }

    def close_gripper_pybullet_style_recordable(
        self,
        max_steps=80,
        min_steps=5,
        stable_steps=5,
        width_delta_tolerance=1e-4,
        frame_callback=None,
    ):
        """Close the Panda gripper until its motion becomes stable.

        Keep commanding close while the gripper is moving, and stop once
        the gripper width remains effectively unchanged for a short
        consecutive window.

        This function is independent of the grasp-descent force-stop logic.
        """

        action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )
        action[-1] = 1.0

        max_steps = max(1, int(max_steps))
        min_steps = max(1, int(min_steps))
        stable_steps = max(1, int(stable_steps))

        previous_width = self.get_gripper_width()
        stable_count = 0
        final_width_delta = None
        completed_steps = 0

        for step_index in range(
            1,
            max_steps + 1,
        ):
            self.step(action)
            completed_steps = step_index

            if frame_callback is not None:
                frame_callback(self)

            current_width = self.get_gripper_width()

            width_delta = abs(
                current_width
                - previous_width
            )

            final_width_delta = float(
                width_delta
            )

            if (
                width_delta
                <= float(
                    width_delta_tolerance
                )
            ):
                stable_count += 1
            else:
                stable_count = 0

            previous_width = current_width

            if (
                step_index >= min_steps
                and stable_count >= stable_steps
            ):
                return {
                    "joint_positions": np.asarray(
                        self.robots[0]
                        .get_gripper_joint_positions()
                    ).copy(),
                    "width": float(
                        current_width
                    ),
                    "steps": int(
                        completed_steps
                    ),
                    "stable": True,
                    "stable_steps": int(
                        stable_count
                    ),
                    "width_delta_tolerance": float(
                        width_delta_tolerance
                    ),
                    "final_width_delta": (
                        final_width_delta
                    ),
                    "reached_max_steps": False,
                }

        return {
            "joint_positions": np.asarray(
                self.robots[0]
                .get_gripper_joint_positions()
            ).copy(),
            "width": float(
                self.get_gripper_width()
            ),
            "steps": int(
                completed_steps
            ),
            "stable": False,
            "stable_steps": int(
                stable_count
            ),
            "width_delta_tolerance": float(
                width_delta_tolerance
            ),
            "final_width_delta": (
                final_width_delta
            ),
            "reached_max_steps": True,
        }


    def move_to_ik_safe_rest(
        self,
        arm="right",
        joint_tolerance=0.02,
        qref_speed_rad_per_sec=IK_QREF_SPEED_RAD_PER_SEC,
        frame_callback=None,
        frame_capture_interval=10,
    ):
        """Move to the validated Panda safe-rest joint configuration."""

        return self.move_joints_qref(
            target_joint_positions=(
                IK_SAFE_REST_JOINTS
            ),
            arm=arm,
            qref_speed_rad_per_sec=(
                qref_speed_rad_per_sec
            ),
            joint_tolerance=(
                joint_tolerance
            ),
            stop_on_table_contact=False,
            frame_callback=frame_callback,
            frame_capture_interval=(
                frame_capture_interval
            ),
        )

    def command_gripper_recordable(
        self,
        command,
        steps=30,
        frame_callback=None,
    ):
        """Apply a gripper command and optionally record every control step."""

        action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        action[-1] = np.clip(
            command,
            -1.0,
            1.0,
        )

        for _ in range(
            int(steps)
        ):
            self.step(action)

            if frame_callback is not None:
                frame_callback(self)

        return {
            "joint_positions": np.asarray(
                self.robots[0]
                .get_gripper_joint_positions()
            ).copy(),
            "width": self.get_gripper_width(),
        }

    def execute_grasp_pose_ik(
        self,
        grasp_pose,
        pregrasp_height=0.20,
        lift_distance=0.12,
        grasp_depth_offset=0.015,
        waypoint_spacing=IK_CARTESIAN_WAYPOINT_SPACING,
        qref_speed_rad_per_sec=IK_QREF_SPEED_RAD_PER_SEC,
        frame_callback=None,
        frame_capture_interval=10,
        stop_after_pregrasp=False,
        stop_after_grasp_pose=False,
    ):
        """Execute one 7D grasp with IK + JOINT_POSITION.

        The pregrasp position preserves grasp X-Y and orientation while
        raising the target along world +Z. The subsequent straight_move_ik()
        descends through Cartesian waypoints while preserving orientation.

        Optional diagnostic flags can stop execution after pregrasp or after
        reaching the grasp pose.
        """

        grasp_pose = np.asarray(
            grasp_pose,
            dtype=np.float64,
        ).reshape(7)

        grasp_eef_pose = (
            self.grasp_tip_pose_to_eef_pose(
                grasp_pose
            )
        )

        grasp_rotation = (
            grasp_eef_pose[:3, :3]
        )
        approach_direction = (
            grasp_rotation[:, 2]
        )

        grasp_eef_pose = (
            grasp_eef_pose.copy()
        )
        grasp_eef_pose[:3, 3] -= (
            float(grasp_depth_offset)
            * approach_direction
        )

        # Pregrasp offset is applied along world +Z, not along
        # the local approach axis.
        pregrasp_pose = (
            grasp_eef_pose.copy()
        )
        pregrasp_pose[2, 3] += float(
            pregrasp_height
        )

        lift_pose = (
            grasp_eef_pose.copy()
        )
        lift_pose[2, 3] += float(
            lift_distance
        )

        open_result = (
            self.command_gripper_recordable(
                command=-1.0,
                steps=30,
                frame_callback=(
                    frame_callback
                ),
            )
        )

        # Solve pregrasp directly from the current/home joint state.
        # The optional safe-rest posture is not inserted before this motion.
        safe_rest_result = None

        pregrasp_result = (
            self.move_pose_ik(
                target_pose=pregrasp_pose,
                joint_tolerance=0.02,
                qref_speed_rad_per_sec=(
                    qref_speed_rad_per_sec
                ),
                stop_on_table_contact=True,
                frame_callback=(
                    frame_callback
                ),
                frame_capture_interval=(
                    frame_capture_interval
                ),
            )
        )

        if not pregrasp_result["success"]:
            return {
                "success": False,
                "failed_phase": "pregrasp",
                "open": open_result,
                "safe_rest": (
                    safe_rest_result
                ),
                "pregrasp": (
                    pregrasp_result
                ),
                "grasp": None,
                "close": None,
                "lift": None,
                "grasp_eef_pose": (
                    grasp_eef_pose
                ),
                "pregrasp_pose": (
                    pregrasp_pose
                ),
                "lift_pose": lift_pose,
            }

        if stop_after_pregrasp:
            return {
                "success": True,
                "stopped_for_diagnostic": (
                    "pregrasp"
                ),
                "failed_phase": None,
                "open": open_result,
                "safe_rest": (
                    safe_rest_result
                ),
                "pregrasp": (
                    pregrasp_result
                ),
                "grasp": None,
                "close": None,
                "lift": None,
                "grasp_eef_pose": (
                    grasp_eef_pose
                ),
                "pregrasp_pose": (
                    pregrasp_pose
                ),
                "lift_pose": lift_pose,
            }

        grasp_result = (
            self.straight_move_ik(
                target_pose=(
                    grasp_eef_pose
                ),
                waypoint_spacing=(
                    waypoint_spacing
                ),
                joint_tolerance=0.012,
                qref_speed_rad_per_sec=(
                    qref_speed_rad_per_sec
                ),
                stop_on_table_contact=True,
                force_stop_enabled=True,
                force_stop_threshold=PYBULLET_STYLE_GRASP_FORCE_THRESHOLD,
                force_stop_consecutive_steps=GRASP_FORCE_STOP_CONSECUTIVE_STEPS,
                frame_callback=(
                    frame_callback
                ),
                frame_capture_interval=(
                    frame_capture_interval
                ),
            )
        )

        if not grasp_result["success"]:
            return {
                "success": False,
                "failed_phase": "grasp",
                "open": open_result,
                "safe_rest": (
                    safe_rest_result
                ),
                "pregrasp": (
                    pregrasp_result
                ),
                "grasp": grasp_result,
                "close": None,
                "lift": None,
                "grasp_eef_pose": (
                    grasp_eef_pose
                ),
                "pregrasp_pose": (
                    pregrasp_pose
                ),
                "lift_pose": lift_pose,
            }

        if stop_after_grasp_pose:
            return {
                "success": True,
                "stopped_for_diagnostic": (
                    "grasp_pose"
                ),
                "failed_phase": None,
                "open": open_result,
                "safe_rest": (
                    safe_rest_result
                ),
                "pregrasp": (
                    pregrasp_result
                ),
                "grasp": grasp_result,
                "close": None,
                "lift": None,
                "grasp_eef_pose": (
                    grasp_eef_pose
                ),
                "pregrasp_pose": (
                    pregrasp_pose
                ),
                "lift_pose": lift_pose,
            }

        close_result = (
            self.close_gripper_pybullet_style_recordable(
                max_steps=80,
                min_steps=5,
                stable_steps=5,
                width_delta_tolerance=1e-4,
                frame_callback=(
                    frame_callback
                ),
            )
        )

        print(
            "Gripper close:",
            {
                "steps": close_result["steps"],
                "stable": close_result["stable"],
                "width": close_result["width"],
                "final_width_delta": (
                    close_result["final_width_delta"]
                ),
                "reached_max_steps": (
                    close_result["reached_max_steps"]
                ),
            },
        )

        lift_result = (
            self.straight_move_ik(
                target_pose=lift_pose,
                waypoint_spacing=(
                    waypoint_spacing
                ),
                joint_tolerance=0.012,
                qref_speed_rad_per_sec=(
                    qref_speed_rad_per_sec
                ),
                stop_on_table_contact=False,
                frame_callback=(
                    frame_callback
                ),
                frame_capture_interval=(
                    frame_capture_interval
                ),
            )
        )

        return {
            "success": bool(
                lift_result["success"]
            ),
            "failed_phase": (
                None
                if lift_result["success"]
                else "lift"
            ),
            "open": open_result,
            "safe_rest": (
                safe_rest_result
            ),
            "pregrasp": (
                pregrasp_result
            ),
            "grasp": grasp_result,
            "close": close_result,
            "lift": lift_result,
            "grasp_eef_pose": (
                grasp_eef_pose
            ),
            "pregrasp_pose": (
                pregrasp_pose
            ),
            "lift_pose": lift_pose,
            "final_gripper_width": (
                self.get_gripper_width()
            ),
        }

    def move_to_pose_smooth_trajectory(
        self,
        target_pose,
        arm="right",
        position_tolerance=0.008,
        orientation_tolerance=np.deg2rad(4.0),
        gripper_command=0.0,
        position_gain=0.5,
        orientation_gain=0.5,
        nominal_linear_speed=0.12,
        nominal_angular_speed=np.deg2rad(40.0),
        min_trajectory_steps=20,
        max_trajectory_steps=140,
        final_settle_steps=80,
        stop_on_table_contact=False,
    ):
        """Track a smooth Cartesian pose trajectory with OSC.

        Instead of commanding the distant final pose immediately, generate
        a quintic time-scaled trajectory from the current EEF pose to the
        target pose. The reference starts and ends with zero nominal
        velocity, reducing the impulsive "rush" caused by a large pose jump.

        Position is interpolated linearly with quintic time scaling.
        Orientation follows the shortest relative rotation with the same
        time scaling.
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float64,
        )

        if target_pose.shape != (4, 4):
            raise ValueError(
                f"target_pose must have shape (4, 4), "
                f"got {target_pose.shape}"
            )

        start_pose = self.get_eef_pose(arm=arm)
        start_position = start_pose[:3, 3].copy()
        start_rotation = start_pose[:3, :3].copy()

        target_position = target_pose[:3, 3].copy()
        target_rotation = target_pose[:3, :3].copy()

        translation_distance = float(
            np.linalg.norm(
                target_position - start_position
            )
        )

        relative_rotation = (
            target_rotation @ start_rotation.T
        )
        relative_rotvec = (
            SciPyRotation.from_matrix(
                relative_rotation
            ).as_rotvec()
        )
        rotation_angle = float(
            np.linalg.norm(relative_rotvec)
        )

        linear_duration = (
            translation_distance
            / max(float(nominal_linear_speed), 1e-6)
        )
        angular_duration = (
            rotation_angle
            / max(float(nominal_angular_speed), 1e-6)
        )
        duration = max(
            linear_duration,
            angular_duration,
            1.0 / 20.0,
        )

        trajectory_steps = int(
            np.ceil(duration * 20.0)
        )
        trajectory_steps = int(
            np.clip(
                trajectory_steps,
                int(min_trajectory_steps),
                int(max_trajectory_steps),
            )
        )

        controller = self.robots[0].part_controllers[arm]

        max_position_delta = np.asarray(
            controller.output_max[:3],
            dtype=np.float64,
        )
        max_rotation_delta = np.asarray(
            controller.output_max[3:6],
            dtype=np.float64,
        )

        for trajectory_step in range(
            1,
            trajectory_steps + 1,
        ):
            tau = (
                trajectory_step
                / float(trajectory_steps)
            )

            # Quintic smoothstep:
            # s(0)=0, s(1)=1 and first / second derivatives vanish
            # at both ends.
            s = (
                10.0 * tau**3
                - 15.0 * tau**4
                + 6.0 * tau**5
            )

            reference_position = (
                start_position
                + s
                * (
                    target_position
                    - start_position
                )
            )

            reference_rotation = (
                SciPyRotation.from_rotvec(
                    s * relative_rotvec
                ).as_matrix()
                @ start_rotation
            )

            current_pose = self.get_eef_pose(
                arm=arm
            )
            current_position = (
                current_pose[:3, 3]
            )
            current_rotation = (
                current_pose[:3, :3]
            )

            position_error = (
                reference_position
                - current_position
            )

            rotation_error_matrix = (
                reference_rotation
                @ current_rotation.T
            )
            rotation_error = (
                SciPyRotation.from_matrix(
                    rotation_error_matrix
                ).as_rotvec()
            )

            action = np.zeros(
                self.action_dim,
                dtype=np.float32,
            )

            action[:3] = np.clip(
                position_gain
                * position_error
                / max_position_delta,
                -1.0,
                1.0,
            )

            action[3:6] = np.clip(
                orientation_gain
                * rotation_error
                / max_rotation_delta,
                -1.0,
                1.0,
            )

            action[-1] = np.clip(
                gripper_command,
                -1.0,
                1.0,
            )

            self.step(action)

            if (
                stop_on_table_contact
                and self.has_gripper_table_contact()
            ):
                return {
                    "success": False,
                    "steps": trajectory_step,
                    "failure_reason": "table_contact",
                    "table_contact": True,
                    "trajectory_steps": trajectory_steps,
                }

        # After the smooth reference reaches the final pose, use the normal
        # pose controller only for the remaining small residual error.
        final_result = self.move_to_pose(
            target_pose=target_pose,
            arm=arm,
            max_steps=final_settle_steps,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            gripper_command=gripper_command,
            position_gain=0.30,
            orientation_gain=0.30,
            stop_on_table_contact=stop_on_table_contact,
        )

        return {
            **final_result,
            "trajectory_steps": trajectory_steps,
            "translation_distance": translation_distance,
            "rotation_angle": rotation_angle,
        }

    def move_to_pose(
        self,
        target_pose,
        arm="right",
        max_steps=200,
        position_tolerance=0.008,
        orientation_tolerance=np.deg2rad(3.0),
        gripper_command=0.0,
        position_gain=0.5,
        orientation_gain=0.5,
        stop_on_table_contact=False,
        linear_velocity_tolerance=None,
        angular_velocity_tolerance=None,
        settle_steps=1,
        joint_position_target=None,
        joint_position_tolerance=None,
    ):
        """Move the Panda EEF toward a world-frame 4x4 target pose.

        The OSC controller accepts:
            action[0:3]: normalized delta position
            action[3:6]: normalized axis-angle delta orientation
            action[-1]: gripper command
        """

        target_pose = np.asarray(
            target_pose,
            dtype=np.float64,
        )

        if target_pose.shape != (4, 4):
            raise ValueError(
                f"target_pose must have shape (4, 4), "
                f"got {target_pose.shape}"
            )

        if joint_position_target is not None:
            joint_position_target = np.asarray(
                joint_position_target,
                dtype=np.float64,
            ).reshape(-1)

            current_joint_count = len(
                self.get_arm_joint_positions()
            )

            if len(joint_position_target) != current_joint_count:
                raise ValueError(
                    "joint_position_target size does not match "
                    f"the arm DoF: {len(joint_position_target)} "
                    f"vs {current_joint_count}"
                )

            if joint_position_tolerance is None:
                raise ValueError(
                    "joint_position_tolerance is required when "
                    "joint_position_target is provided"
                )

        target_position = target_pose[:3, 3]
        target_rotation = target_pose[:3, :3]

        controller = self.robots[0].part_controllers[arm]

        settle_steps = max(1, int(settle_steps))
        settled_count = 0

        max_position_delta = np.asarray(
            controller.output_max[:3],
            dtype=np.float64,
        )
        max_rotation_delta = np.asarray(
            controller.output_max[3:6],
            dtype=np.float64,
        )

        for step in range(max_steps):
            current_pose = self.get_eef_pose(arm=arm)
            current_position = current_pose[:3, 3]
            current_rotation = current_pose[:3, :3]

            position_error = target_position - current_position

            # Controller applies:
            # R_goal = R_error @ R_current
            rotation_error_matrix = (
                target_rotation @ current_rotation.T
            )

            rotation_error = SciPyRotation.from_matrix(
                rotation_error_matrix
            ).as_rotvec()

            position_error_norm = float(
                np.linalg.norm(position_error)
            )
            orientation_error_norm = float(
                np.linalg.norm(rotation_error)
            )

            linear_velocity, angular_velocity = (
                self.get_eef_velocity(arm=arm)
            )
            linear_speed = float(np.linalg.norm(linear_velocity))
            angular_speed = float(np.linalg.norm(angular_velocity))

            pose_within_tolerance = (
                position_error_norm <= position_tolerance
                and orientation_error_norm <= orientation_tolerance
            )

            current_joint_positions = self.get_arm_joint_positions()

            if joint_position_target is None:
                joint_position_error = None
                joint_position_error_max = None
                joints_within_tolerance = True
            else:
                joint_position_error = (
                    joint_position_target
                    - current_joint_positions
                )
                joint_position_error_max = float(
                    np.max(np.abs(joint_position_error))
                )
                joints_within_tolerance = (
                    joint_position_error_max
                    <= float(joint_position_tolerance)
                )

            velocity_within_tolerance = True
            if linear_velocity_tolerance is not None:
                velocity_within_tolerance &= (
                    linear_speed <= float(linear_velocity_tolerance)
                )
            if angular_velocity_tolerance is not None:
                velocity_within_tolerance &= (
                    angular_speed <= float(angular_velocity_tolerance)
                )

            if (
                pose_within_tolerance
                and velocity_within_tolerance
                and joints_within_tolerance
            ):
                settled_count += 1
            else:
                settled_count = 0

            if settled_count >= settle_steps:
                return {
                    "success": True,
                    "steps": step,
                    "failure_reason": None,
                    "table_contact": False,
                    "final_pose": current_pose,
                    "position_error": position_error,
                    "orientation_error": rotation_error,
                    "position_error_norm": position_error_norm,
                    "orientation_error_norm": orientation_error_norm,
                    "linear_speed": linear_speed,
                    "angular_speed": angular_speed,
                    "settled_steps": settled_count,
                    "joint_positions": current_joint_positions,
                    "joint_position_error": joint_position_error,
                    "joint_position_error_max": joint_position_error_max,
                }

            action = np.zeros(
                self.action_dim,
                dtype=np.float32,
            )

            action[:3] = np.clip(
                position_gain
                * position_error
                / max_position_delta,
                -1.0,
                1.0,
            )

            action[3:6] = np.clip(
                orientation_gain
                * rotation_error
                / max_rotation_delta,
                -1.0,
                1.0,
            )

            action[-1] = np.clip(
                gripper_command,
                -1.0,
                1.0,
            )

            self.step(action)

            if (
                stop_on_table_contact
                and self.has_gripper_table_contact()
            ):
                contact_pose = self.get_eef_pose(
                    arm=arm
                )

                contact_position_error = (
                    target_position
                    - contact_pose[:3, 3]
                )

                contact_rotation_error = (
                    target_rotation
                    @ contact_pose[:3, :3].T
                )

                contact_orientation_error = (
                    SciPyRotation.from_matrix(
                        contact_rotation_error
                    ).as_rotvec()
                )

                return {
                    "success": False,
                    "steps": step + 1,
                    "failure_reason": "table_contact",
                    "table_contact": True,
                    "final_pose": contact_pose,
                    "position_error": contact_position_error,
                    "orientation_error": contact_orientation_error,
                    "position_error_norm": float(
                        np.linalg.norm(
                            contact_position_error
                        )
                    ),
                    "orientation_error_norm": float(
                        np.linalg.norm(
                            contact_orientation_error
                        )
                    ),
                }

        final_pose = self.get_eef_pose(arm=arm)
        final_position_error = (
            target_position - final_pose[:3, 3]
        )

        final_rotation_error_matrix = (
            target_rotation @ final_pose[:3, :3].T
        )
        final_orientation_error = SciPyRotation.from_matrix(
            final_rotation_error_matrix
        ).as_rotvec()

        position_error_norm = float(
            np.linalg.norm(final_position_error)
        )
        orientation_error_norm = float(
            np.linalg.norm(final_orientation_error)
        )

        final_linear_velocity, final_angular_velocity = (
            self.get_eef_velocity(arm=arm)
        )
        final_linear_speed = float(
            np.linalg.norm(final_linear_velocity)
        )
        final_angular_speed = float(
            np.linalg.norm(final_angular_velocity)
        )

        final_joint_positions = self.get_arm_joint_positions()

        if joint_position_target is None:
            final_joint_position_error = None
            final_joint_position_error_max = None
        else:
            final_joint_position_error = (
                joint_position_target
                - final_joint_positions
            )
            final_joint_position_error_max = float(
                np.max(np.abs(final_joint_position_error))
            )

        return {
            "success": (
                position_error_norm <= position_tolerance
                and orientation_error_norm <= orientation_tolerance
            ),
            "steps": max_steps,
            "failure_reason": "timeout",
            "table_contact": False,
            "final_pose": final_pose,
            "position_error": final_position_error,
            "orientation_error": final_orientation_error,
            "position_error_norm": position_error_norm,
            "orientation_error_norm": orientation_error_norm,
            "linear_speed": final_linear_speed,
            "angular_speed": final_angular_speed,
            "settled_steps": settled_count,
            "joint_positions": final_joint_positions,
            "joint_position_error": final_joint_position_error,
            "joint_position_error_max": final_joint_position_error_max,
        }

    def get_eef_position(self, arm="right"):
        """Return the end-effector site position in world coordinates."""

        site_ids = self.robots[0].eef_site_id

        if isinstance(site_ids, dict):
            if arm not in site_ids:
                raise KeyError(f"Unknown arm: {arm}")
            site_id = site_ids[arm]
        else:
            site_id = site_ids

        return self.sim.data.site_xpos[site_id].copy()

    def move_to_position(
        self,
        target_position,
        arm="right",
        max_steps=100,
        position_tolerance=0.005,
        gripper_command=0.0,
        gain=0.5,
    ):
        """Move the end effector toward a world-frame target position.

        The current OSC controller uses delta position commands.
        Orientation commands remain zero, so the current orientation is held.
        """

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).reshape(3)

        arm_controller = self.robots[0].part_controllers[arm]
        max_position_delta = np.asarray(
            arm_controller.output_max[:3],
            dtype=np.float64,
        )

        for step in range(max_steps):
            current_position = self.get_eef_position(arm=arm)
            position_error = target_position - current_position

            if np.linalg.norm(position_error) <= position_tolerance:
                return {
                    "success": True,
                    "steps": step,
                    "final_position": current_position,
                    "position_error": position_error,
                }

            action = np.zeros(
                self.action_dim,
                dtype=np.float32,
            )

            delta_command = (
                gain * position_error / max_position_delta
            )

            action[:3] = np.clip(
                delta_command,
                -1.0,
                1.0,
            )

            # Keep action[3:6] at zero so orientation is not actively changed.
            action[-1] = np.clip(
                gripper_command,
                -1.0,
                1.0,
            )

            self.step(action)

        final_position = self.get_eef_position(arm=arm)
        final_error = target_position - final_position

        return {
            "success": (
                np.linalg.norm(final_error)
                <= position_tolerance
            ),
            "steps": max_steps,
            "final_position": final_position,
            "position_error": final_error,
        }


    def safe_return_home(
        self,
        home_pose,
        home_joint_positions=None,
        arm="right",
        safe_height=1.02,
        gripper_command=-1.0,
        joint_position_tolerance=0.10,
    ):
        """Return home through two safe Cartesian waypoints.

        Sequence:
            1. Lift vertically at the current XY position.
            2. Move horizontally above the home XY position.
            3. Descend to the complete saved home pose.

        The current robosuite controller is OSC pose control, so this
        method avoids one large direct Cartesian motion.
        """

        home_pose = np.asarray(
            home_pose,
            dtype=np.float64,
        )

        if home_pose.shape != (4, 4):
            raise ValueError(
                "home_pose must have shape (4, 4), "
                f"got {home_pose.shape}"
            )

        current_position = self.get_eef_position(
            arm=arm
        )

        safe_height = max(
            float(safe_height),
            float(current_position[2]),
            float(home_pose[2, 3]),
        )

        # Phase 1: lift vertically without changing orientation.
        lift_position = current_position.copy()
        lift_position[2] = safe_height

        lift_result = self.move_to_position(
            target_position=lift_position,
            arm=arm,
            max_steps=160,
            position_tolerance=0.012,
            gripper_command=gripper_command,
            gain=0.25,
        )

        if not lift_result["success"]:
            return {
                "success": False,
                "failed_phase": "safe_lift",
                "safe_lift": lift_result,
                "safe_horizontal": None,
                "final_home": None,
            }

        # Phase 2: move horizontally to a point above home.
        above_home_position = home_pose[:3, 3].copy()
        above_home_position[2] = safe_height

        horizontal_result = self.move_to_position(
            target_position=above_home_position,
            arm=arm,
            max_steps=200,
            position_tolerance=0.015,
            gripper_command=gripper_command,
            gain=0.22,
        )

        if not horizontal_result["success"]:
            return {
                "success": False,
                "failed_phase": "safe_horizontal",
                "safe_lift": lift_result,
                "safe_horizontal": horizontal_result,
                "final_home": None,
            }

        # Phase 3: restore the complete home position and orientation.
        final_home_result = self.move_to_pose(
            target_pose=home_pose,
            arm=arm,
            max_steps=320,
            position_tolerance=0.015,
            orientation_tolerance=np.deg2rad(6.0),
            gripper_command=gripper_command,
            position_gain=0.20,
            orientation_gain=0.25,
            joint_position_target=home_joint_positions,
            joint_position_tolerance=joint_position_tolerance,
        )

        return {
            "success": bool(final_home_result["success"]),
            "failed_phase": (
                None
                if final_home_result["success"]
                else "final_home"
            ),
            "safe_lift": lift_result,
            "safe_horizontal": horizontal_result,
            "final_home": final_home_result,
        }


    def execute_grasp_pose(
        self,
        grasp_pose,
        approach_distance=0.10,
        lift_distance=0.12,
        grasp_depth_offset=0.015,
        max_steps_per_phase=250,
        position_tolerance=0.008,
        orientation_tolerance=np.deg2rad(4.0),
        grasp_waypoint_spacing=0.02,
    ):
        """Execute a ThinkGrasp 7D world-frame grasp pose.

        Args:
            grasp_pose:
                [x, y, z, qx, qy, qz, qw]

                The final pose returned by ThinkGrasp's
                assign_grasp_pose(). Quaternion ordering is xyzw.

            approach_distance:
                Distance travelled opposite the grasp local +Z axis
                before descending to the final grasp pose.

            lift_distance:
                Vertical world-frame lifting distance after closing.

            grasp_depth_offset:
                Panda-specific compensation applied opposite the grasp
                approaching direction before execution. A positive value
                keeps the gripper palm farther away from the object while
                preserving the original grasp orientation.

            grasp_waypoint_spacing:
                Maximum Cartesian spacing between consecutive targets on
                the straight pregrasp-to-grasp approach. The gripper
                orientation stays equal to the final grasp orientation.
                A value of 0.02 means approximately one waypoint every 2 cm.

        Returns:
            Dictionary containing the result of every motion phase.
        """

        grasp_pose = np.asarray(
            grasp_pose,
            dtype=np.float64,
        ).reshape(7)

        # Convert ThinkGrasp fingertip target into Panda grip_site target.
        grasp_eef_pose = self.grasp_tip_pose_to_eef_pose(
            grasp_pose
        )

        grasp_rotation = grasp_eef_pose[:3, :3]
        approach_direction = grasp_rotation[:, 2]

        # Panda-specific grasp depth compensation.
        # Move the final grip_site target backwards along the approach
        # direction so the object is held between the fingers instead of
        # being pushed into the gripper palm.
        grasp_eef_pose[:3, 3] -= (
            grasp_depth_offset * approach_direction
        )

        # ThinkGrasp's local +Z is the approaching direction.
        # Therefore subtracting it moves the gripper backwards / upwards.
        pregrasp_pose = grasp_eef_pose.copy()
        pregrasp_pose[:3, 3] -= (
            approach_distance * approach_direction
        )

        # Lift vertically in the MuJoCo world frame.
        lift_pose = grasp_eef_pose.copy()
        lift_pose[2, 3] += lift_distance

        self.open_gripper()

        # Large current/home-to-pregrasp motion:
        # follow a quintic Cartesian reference instead of lowering the gain
        # on one distant target. This keeps convergence authority while
        # smoothing acceleration and deceleration.
        pregrasp_result = self.move_to_pose_smooth_trajectory(
            target_pose=pregrasp_pose,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            gripper_command=-1.0,
            position_gain=0.50,
            orientation_gain=0.50,
            nominal_linear_speed=0.12,
            nominal_angular_speed=np.deg2rad(40.0),
            final_settle_steps=max_steps_per_phase,
        )

        if not pregrasp_result["success"]:
            return {
                "success": False,
                "failed_phase": "pregrasp",
                "pregrasp": pregrasp_result,
                "grasp": None,
                "lift": None,
                "grasp_eef_pose": grasp_eef_pose,
                "pregrasp_pose": pregrasp_pose,
                "lift_pose": lift_pose,
                "grasp_depth_offset": float(grasp_depth_offset),
            }

        # Follow the straight pregrasp-to-grasp path continuously.
        #
        # The previous implementation used 2 cm waypoints and required the
        # EEF to slow down / settle at every waypoint. That produced the
        # visible stop-go motion. Here the target reference moves
        # continuously along the same straight approach direction with
        # quintic time scaling: smooth acceleration, continuous motion,
        # smooth deceleration near the final grasp pose.
        #
        # Keep the approach deliberately moderate: do not rush into clutter.
        grasp_result = self.move_to_pose_smooth_trajectory(
            target_pose=grasp_eef_pose,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            gripper_command=-1.0,
            position_gain=0.30,
            orientation_gain=0.25,
            nominal_linear_speed=0.08,
            nominal_angular_speed=np.deg2rad(35.0),
            min_trajectory_steps=20,
            max_trajectory_steps=100,
            final_settle_steps=max_steps_per_phase,
            stop_on_table_contact=True,
        )

        grasp_result = {
            **grasp_result,
            "approach_mode": "continuous_smooth_trajectory",
            "nominal_linear_speed": 0.08,
        }

        # Kept for compatibility with the runner / log format. There are no
        # stop-and-settle intermediate waypoints anymore.
        grasp_waypoint_results = []

        if not grasp_result["success"]:
            return {
                "success": False,
                "failed_phase": "grasp",
                "pregrasp": pregrasp_result,
                "grasp": grasp_result,
                "grasp_waypoints": grasp_waypoint_results,
                "lift": None,
                "grasp_eef_pose": grasp_eef_pose,
                "pregrasp_pose": pregrasp_pose,
                "lift_pose": lift_pose,
                "grasp_depth_offset": float(grasp_depth_offset),
                "grasp_waypoint_spacing": float(
                    grasp_waypoint_spacing
                ),
            }

        self.close_gripper()

        # Lift through the same smooth time-scaled Cartesian reference.
        lift_result = self.move_to_pose_smooth_trajectory(
            target_pose=lift_pose,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            gripper_command=1.0,
            position_gain=0.50,
            orientation_gain=0.50,
            nominal_linear_speed=0.10,
            nominal_angular_speed=np.deg2rad(40.0),
            final_settle_steps=max_steps_per_phase,
        )

        return {
            "success": bool(lift_result["success"]),
            "failed_phase": (
                None if lift_result["success"] else "lift"
            ),
            "pregrasp": pregrasp_result,
            "grasp": grasp_result,
            "grasp_waypoints": grasp_waypoint_results,
            "lift": lift_result,
            "grasp_eef_pose": grasp_eef_pose,
            "pregrasp_pose": pregrasp_pose,
            "lift_pose": lift_pose,
            "grasp_depth_offset": float(grasp_depth_offset),
            "grasp_waypoint_spacing": float(
                grasp_waypoint_spacing
            ),
            "final_gripper_width": self.get_gripper_width(),
        }

    def grasp_tip_pose_to_eef_pose(
        self,
        grasp_tip_pose,
        tip_offset=0.0044,
    ):
        """Convert a ThinkGrasp world-frame tip pose to Panda grip_site pose.

        Args:
            grasp_tip_pose:
                [x, y, z, qx, qy, qz, qw]

                ThinkGrasp final grasp pose in the MuJoCo world frame.
                Position represents the gripper fingertip plane.
                Quaternion ordering is xyzw.

            tip_offset:
                Panda fingertip plane position relative to grip_site.
                Measured from the current Panda model as 4.4 mm
                along local +Z.

        Returns:
            4x4 world-frame target pose of Panda grip_site.
        """

        grasp_tip_pose = np.asarray(
            grasp_tip_pose,
            dtype=np.float64,
        ).reshape(7)

        tip_position = grasp_tip_pose[:3]
        tip_quaternion_xyzw = grasp_tip_pose[3:]

        tip_rotation = quat2mat(
            tip_quaternion_xyzw
        )

        local_eef_to_tip = np.array(
            [0.0, 0.0, tip_offset],
            dtype=np.float64,
        )

        eef_pose = np.eye(4, dtype=np.float64)
        eef_pose[:3, :3] = tip_rotation

        # tip = eef + R @ local_eef_to_tip
        # therefore eef = tip - R @ local_eef_to_tip
        eef_pose[:3, 3] = (
            tip_position
            - tip_rotation @ local_eef_to_tip
        )

        return eef_pose

    def grasp_pose_to_eef_pose(
        self,
        grasp_pose,
        grasp_center_offset=0.0036,
    ):
        """Convert a GraspNet world-frame grasp pose to Panda EEF pose.

        Args:
            grasp_pose:
                [x, y, z, qx, qy, qz, qw]

                The position represents the desired grasp center.
                The quaternion uses xyzw ordering.

            grasp_center_offset:
                Distance from Panda grip_site to the center between
                the two finger pads. Measured in the current model
                as approximately 3.6 mm along local -Z.

        Returns:
            4x4 world-frame pose of Panda grip_site.
        """

        grasp_pose = np.asarray(
            grasp_pose,
            dtype=np.float64,
        ).reshape(7)

        grasp_position = grasp_pose[:3]
        grasp_quaternion_xyzw = grasp_pose[3:]

        grasp_rotation = quat2mat(
            grasp_quaternion_xyzw
        )

        grasp_transform = np.eye(
            4,
            dtype=np.float64,
        )
        grasp_transform[:3, :3] = grasp_rotation
        grasp_transform[:3, 3] = grasp_position

        # grasp_center = eef + R @ [0, 0, -0.0036]
        # Therefore:
        # eef = grasp_center + R @ [0, 0, +0.0036]
        local_center_to_eef = np.array(
            [0.0, 0.0, grasp_center_offset],
            dtype=np.float64,
        )

        eef_transform = grasp_transform.copy()
        eef_transform[:3, 3] = (
            grasp_position
            + grasp_rotation @ local_center_to_eef
        )

        return eef_transform

    def grasp_object(
        self,
        obj_id,
        approach_height=0.15,
        grasp_height_offset=None,
        lift_height=0.10,
        success_lift_threshold=0.05,
    ):
        """Execute a simple vertical grasp on one object.

        This is the first validated MuJoCo grasping prototype:
        1. open gripper
        2. move above object
        3. descend to grasp height
        4. close gripper
        5. lift object
        6. verify object lift distance
        """

        if obj_id not in self.body_id_to_name:
            raise KeyError(f"Unknown object body id: {obj_id}")

        object_name = self.body_id_to_name[obj_id]

        # Open the gripper and let the scene settle before reading the object pose.
        open_result = self.open_gripper(steps=30)

        # Read the object position only after settling.
        object_start = self.sim.data.body_xpos[obj_id].copy()

        # Simple geometric vertical-grasp heuristic:
        # grasp height = half object height - 5 mm.
        # The GraspNet path uses its predicted grasp pose instead.
        if grasp_height_offset is None:
            _, _, dimensions = self.obj_info(obj_id)
            object_height = float(dimensions[2])
            grasp_height_offset = object_height / 2.0 - 0.005

        approach_target = object_start.copy()
        approach_target[2] += approach_height

        approach_result = self.move_to_position(
            target_position=approach_target,
            max_steps=180,
            position_tolerance=0.01,
            gripper_command=-1.0,
            gain=0.5,
        )

        if not approach_result["success"]:
            return {
                "success": False,
                "stage": "approach",
                "object_name": object_name,
                "object_id": obj_id,
                "approach_result": approach_result,
            }

        grasp_target = object_start.copy()
        grasp_target[2] += grasp_height_offset

        descend_result = self.move_to_position(
            target_position=grasp_target,
            max_steps=180,
            position_tolerance=0.005,
            gripper_command=-1.0,
            gain=0.2,
        )

        if not descend_result["success"]:
            return {
                "success": False,
                "stage": "descend",
                "object_name": object_name,
                "object_id": obj_id,
                "approach_result": approach_result,
                "descend_result": descend_result,
            }

        object_before_close = self.sim.data.body_xpos[obj_id].copy()
        close_result = self.close_gripper(steps=50)
        object_after_close = self.sim.data.body_xpos[obj_id].copy()

        eef_before_lift = self.get_eef_position()
        lift_target = eef_before_lift.copy()
        lift_target[2] += lift_height

        lift_result = self.move_to_position(
            target_position=lift_target,
            max_steps=200,
            position_tolerance=0.008,
            gripper_command=1.0,
            gain=0.35,
        )

        object_after_lift = self.sim.data.body_xpos[obj_id].copy()
        eef_after_lift = self.get_eef_position()

        object_lift_distance = float(
            object_after_lift[2] - object_before_close[2]
        )

        eef_lift_distance = float(
            eef_after_lift[2] - eef_before_lift[2]
        )

        success = (
            lift_result["success"]
            and object_lift_distance >= success_lift_threshold
        )

        return {
            "success": success,
            "stage": "finished",
            "object_name": object_name,
            "object_id": obj_id,
            "grasp_height_offset": float(grasp_height_offset),
            "open_result": open_result,
            "approach_result": approach_result,
            "descend_result": descend_result,
            "close_result": close_result,
            "lift_result": lift_result,
            "object_start": object_start,
            "object_before_close": object_before_close,
            "object_after_close": object_after_close,
            "object_after_lift": object_after_lift,
            "object_lift_distance": object_lift_distance,
            "eef_lift_distance": eef_lift_distance,
            "final_gripper_width": self.get_gripper_width(),
            "final_eef_object_difference": (
                eef_after_lift - object_after_lift
            ),
        }

    def get_object_poses(self):
        poses = {}

        for name, body_id in self.object_body_ids.items():
            poses[name] = {
                "position": self.sim.data.body_xpos[body_id].copy(),
                "quaternion_wxyz": self.sim.data.body_xquat[body_id].copy(),
            }

        return poses

    def obj_info(self, obj_id):
        """Return object position, quaternion and conservative dimensions."""

        if obj_id not in self.body_id_to_name:
            raise KeyError(f"Unknown object body id: {obj_id}")

        position = self.sim.data.body_xpos[obj_id].copy()
        quaternion_wxyz = self.sim.data.body_xquat[obj_id].copy()
        quaternion_xyzw = convert_quat(
            quaternion_wxyz,
            to="xyzw",
        )

        # The GraspNet path does not depend on this heuristic size.
        # Keep grasp_object() backward compatible by returning a conservative
        # bounding diameter derived from all geoms belonging to this object's
        # body subtree.
        object_geom_radii = []

        for geom_id in range(self.sim.model.ngeom):
            body_id = int(
                self.sim.model.geom_bodyid[geom_id]
            )
            current_body_id = body_id

            while current_body_id > 0:
                if current_body_id == obj_id:
                    object_geom_radii.append(
                        float(
                            self.sim.model.geom_rbound[
                                geom_id
                            ]
                        )
                    )
                    break

                current_body_id = int(
                    self.sim.model.body_parentid[
                        current_body_id
                    ]
                )

        if object_geom_radii:
            diameter = 2.0 * max(object_geom_radii)
        else:
            diameter = 0.08

        dimensions = np.array(
            [diameter, diameter, diameter],
            dtype=np.float64,
        )

        return position, quaternion_xyzw, dimensions


    def get_true_object_pose(self, obj_id):
        """Return one object's world-frame 4x4 pose."""

        if obj_id not in self.body_id_to_name:
            raise KeyError(f"Unknown object body id: {obj_id}")

        position = self.sim.data.body_xpos[obj_id].copy()
        quaternion_wxyz = self.sim.data.body_xquat[obj_id].copy()
        quaternion_xyzw = convert_quat(
            quaternion_wxyz,
            to="xyzw",
        )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat2mat(quaternion_xyzw)
        transform[:3, 3] = position

        return transform

    def get_task_info(self):
        """Return current language goal and target object information."""

        return {
            "lang_goal": self.lang_goal,
            "target_obj_names": list(self.target_obj_names),
            "target_obj_ids": list(self.target_obj_ids),
            "rigid_obj_ids": list(self.obj_ids["rigid"]),
        }

    def get_true_object_poses(self):
        """Return world-frame 4x4 poses indexed by MuJoCo body id."""

        transforms = {}

        for body_id in self.body_id_to_name:
            position = self.sim.data.body_xpos[body_id].copy()
            quaternion_wxyz = self.sim.data.body_xquat[body_id].copy()
            quaternion_xyzw = convert_quat(
                quaternion_wxyz,
                to="xyzw",
            )

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = quat2mat(quaternion_xyzw)
            transform[:3, 3] = position

            transforms[body_id] = transform

        return transforms


    def get_camera_intrinsics(
        self,
        camera_name="agentview",
        width=640,
        height=480,
    ):
        """Return the 3x3 camera intrinsic matrix."""

        return get_camera_intrinsic_matrix(
            self.sim,
            camera_name,
            height,
            width,
        ).astype(np.float64)

    def get_camera_extrinsics(
        self,
        camera_name="agentview",
    ):
        """Return the 4x4 camera-to-world transformation matrix."""

        return get_camera_extrinsic_matrix(
            self.sim,
            camera_name,
        ).astype(np.float64)


    def get_pointcloud(
        self,
        depth,
        camera_name="agentview",
    ):
        """Convert a metric depth image to an H x W x 3 world-frame point cloud."""

        depth = np.asarray(depth, dtype=np.float32)

        if depth.ndim != 2:
            raise ValueError(
                f"depth must have shape (H, W), got {depth.shape}"
            )

        height, width = depth.shape

        K = self.get_camera_intrinsics(
            camera_name=camera_name,
            width=width,
            height=height,
        )

        camera_to_world = self.get_camera_extrinsics(
            camera_name=camera_name,
        )

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        u, v = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
        )

        z = depth.astype(np.float64)

        # MuJoCo image rows use the opposite vertical convention to camera projection.
        v_corrected = height - 1 - v

        x = (u - cx) * z / fx
        y = (v_corrected - cy) * z / fy

        camera_points = np.stack(
            [x, y, z, np.ones_like(z)],
            axis=-1,
        )

        world_points = camera_points @ camera_to_world.T

        return world_points[..., :3].astype(np.float32)


    def get_camera_data(
        self,
        camera_name="agentview",
        width=640,
        height=480,
    ):
        """Return all camera data required by the ThinkGrasp pipeline."""

        color, depth, segmentation = self.render_camera({
            "camera_name": camera_name,
            "width": width,
            "height": height,
        })

        pointcloud = self.get_pointcloud(
            depth,
            camera_name=camera_name,
        )

        intrinsics = self.get_camera_intrinsics(
            camera_name=camera_name,
            width=width,
            height=height,
        )

        extrinsics = self.get_camera_extrinsics(
            camera_name=camera_name,
        )

        return {
            "color": color,
            "depth": depth,
            "segmentation": segmentation,
            "pointcloud": pointcloud,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def render_camera(self, config):
        """Render RGB, metric depth, and body-id segmentation."""

        if isinstance(config, str):
            camera_name = config
            width = 640
            height = 480
        elif isinstance(config, dict):
            camera_name = config.get("camera_name", config.get("name", "agentview"))
            if "image_size" in config:
                height, width = config["image_size"]
            else:
                width = int(config.get("width", 640))
                height = int(config.get("height", 480))
        else:
            raise TypeError("config must be a camera name or dict")

        color = self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
            depth=False,
            segmentation=False,
        )

        seg_raw, depth_raw = self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
            depth=True,
            segmentation=True,
        )

        depth = get_real_depth_map(self.sim, depth_raw).astype(np.float32)

        seg_raw = np.asarray(seg_raw, dtype=np.int32)
        obj_types = seg_raw[..., 0]
        obj_ids = seg_raw[..., 1]

        segmentation = np.zeros(obj_ids.shape, dtype=np.int32)
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        geom_mask = obj_types == geom_type

        if np.any(geom_mask):
            geom_ids = obj_ids[geom_mask]
            valid = geom_ids >= 0
            body_ids = np.zeros_like(geom_ids, dtype=np.int32)
            body_ids[valid] = self.sim.model.geom_bodyid[geom_ids[valid]]
            segmentation[geom_mask] = body_ids

        return color, depth, segmentation


def _draw_crop_rectangle(
    image,
    crop_xyxy,
    color=(180, 0, 255),
    thickness=5,
):
    """Draw the confirmed perception crop for human workspace inspection."""

    output = np.asarray(image).copy()

    x1, y1, x2, y2 = [
        int(value)
        for value in np.asarray(crop_xyxy).reshape(4)
    ]

    x1 = int(np.clip(x1, 0, output.shape[1] - 1))
    y1 = int(np.clip(y1, 0, output.shape[0] - 1))
    x2 = int(np.clip(x2, x1 + 1, output.shape[1]))
    y2 = int(np.clip(y2, y1 + 1, output.shape[0]))

    output[y1:y1 + thickness, x1:x2] = color
    output[y2 - thickness:y2, x1:x2] = color
    output[y1:y2, x1:x1 + thickness] = color
    output[y1:y2, x2 - thickness:x2] = color

    return output


def main():
    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

    env = ThinkGraspMinimalEnv(
        controller_configs=controller_config,
    )

    try:
        print("Custom environment created successfully")
        print("Objects:", env.get_object_poses())

        if hasattr(env, "perception_crop_center_pixel"):
            print(
                "Perception crop center pixel:",
                env.perception_crop_center_pixel,
            )

        print(
            "Perception workspace pixel crop:",
            FIXED_PERCEPTION_CROP_XYXY,
        )
        print(
            "Compact drop world bounds:",
            env.object_drop_world_bounds,
        )
        print(
            "Perception workspace world bounds:",
            env._get_perception_workspace_world_bounds(),
        )

        if hasattr(env, "initial_drop_results"):
            print(
                "Initial GSO workspace-drop results:",
                env.initial_drop_results,
            )

        workspace_preview_dir = Path("workspace_preview")
        workspace_preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        topview_rgb, topview_depth, topview_segmentation = (
            env.render_camera(
                {
                    "camera_name": "topview",
                    "width": 640,
                    "height": 640,
                }
            )
        )

        (
            front_oblique_rgb,
            front_oblique_depth,
            front_oblique_segmentation,
        ) = env.render_camera(
            {
                "camera_name": env.front_oblique_camera_name,
                "width": 640,
                "height": 480,
            }
        )

        (
            right_oblique_rgb,
            right_oblique_depth,
            right_oblique_segmentation,
        ) = env.render_camera(
            {
                "camera_name": env.right_oblique_camera_name,
                "width": 640,
                "height": 480,
            }
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = [
            int(value)
            for value in FIXED_PERCEPTION_CROP_XYXY
        ]

        perception_crop = topview_rgb[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ].copy()

        perception_with_box = _draw_crop_rectangle(
            image=topview_rgb,
            crop_xyxy=FIXED_PERCEPTION_CROP_XYXY,
        )

        imageio.imwrite(
            workspace_preview_dir / "topview_full.png",
            topview_rgb,
        )

        imageio.imwrite(
            workspace_preview_dir / "perception_crop.png",
            perception_crop,
        )

        imageio.imwrite(
            workspace_preview_dir / "perception_with_box.png",
            perception_with_box,
        )

        imageio.imwrite(
            workspace_preview_dir / "front_oblique_25deg_rgb.png",
            front_oblique_rgb,
        )

        imageio.imwrite(
            workspace_preview_dir / "right_oblique_25deg_rgb.png",
            right_oblique_rgb,
        )

        right_oblique_depth_normalized = (
            right_oblique_depth
            - np.nanmin(right_oblique_depth)
        )
        depth_range = float(
            np.nanmax(right_oblique_depth_normalized)
        )

        if depth_range > 0.0:
            right_oblique_depth_normalized /= depth_range

        imageio.imwrite(
            workspace_preview_dir / "right_oblique_25deg_depth.png",
            np.clip(
                right_oblique_depth_normalized * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8),
        )

        print(
            "Topview RGB:",
            topview_rgb.shape,
            topview_rgb.dtype,
        )
        print(
            "Topview depth:",
            topview_depth.shape,
            topview_depth.dtype,
        )
        print(
            "Topview segmentation IDs:",
            np.unique(topview_segmentation),
        )
        print(
            "Right oblique camera position:",
            env.right_oblique_camera_position,
        )
        print(
            "Oblique camera target:",
            env.oblique_camera_target,
        )
        print(
            "Oblique downward pitch degrees:",
            env.oblique_camera_pitch_deg,
        )
        print(
            "Right oblique RGB:",
            right_oblique_rgb.shape,
            right_oblique_rgb.dtype,
        )
        print(
            "Right oblique depth:",
            right_oblique_depth.shape,
            right_oblique_depth.dtype,
        )
        print(
            "Right oblique segmentation IDs:",
            np.unique(right_oblique_segmentation),
        )

        for _ in range(10):
            env.step(
                np.zeros(env.action_dim, dtype=np.float32)
            )

        print("Physics stepping successful")
        print(
            "Saved:",
            workspace_preview_dir / "topview_full.png",
        )
        print(
            "Saved:",
            workspace_preview_dir / "perception_crop.png",
        )
        print(
            "Saved:",
            workspace_preview_dir / "perception_with_box.png",
        )
        print(
            "Saved:",
            workspace_preview_dir / "right_oblique_25deg_rgb.png",
        )
        print(
            "Saved:",
            workspace_preview_dir / "right_oblique_25deg_depth.png",
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
