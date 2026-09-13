export const SETTINGS_NAV = [
  {
    id: "scene",
    title: "Scene",
    blurb: "Table, objects, room, and lights used when a scene is generated.",
  },
  {
    id: "cameras",
    title: "Cameras",
    blurb: "Scene views, mounts, and the shared policy slot map used by SFT, GRPO, and Inference.",
  },
  {
    id: "language",
    title: "Language",
    blurb: "Instruction templates for generated scenes: Collect, GRPO, Eval, and Inference.",
  },
  {
    id: "compute",
    title: "Compute",
    blurb: "Render GPU and the device for ACT / SmolVLA training and inference. Pick the checkpoint on the run page.",
  },
  {
    id: "training",
    title: "Training",
    blurb: "Optimizer, cosine scheduler, AMP. Used by SmolVLA SFT and ACT SFT.",
  },
  {
    id: "rollout",
    title: "Rollout",
    blurb: "Episode length and control rate for Inference, GRPO, and Eval. Camera size is the policy render resolution and the scene offscreen buffer.",
  },
  {
    id: "grpo",
    title: "GRPO",
    blurb: "Online Flow-SDE GRPO. Separate from SFT. Uses the Rewards section to score each group of rollouts.",
  },
  {
    id: "eval",
    title: "Eval",
    blurb: "Defaults for the Eval page: valset size and ODE rollout. SmolVLA only.",
  },
  {
    id: "rewards",
    title: "Rewards",
    blurb: "Success and scoring for GRPO, Eval, and Inference tests. Not used by SFT.",
  },
];

export const SETTINGS_GROUPS = {
  scene: [
    {
      id: "objects",
      title: "Objects",
      fields: [
        { path: ["scene"], type: "object_catalog", labelsPath: ["language", "target_labels"], label: "Objects", help: "One list. Target is the pick object. Distractor is an extra object on the table. Rename is used in the instruction." },
        { path: ["scene", "distractors", "min"], type: "number", label: "Distractors, min", help: "Lower bound on extra objects on the table." },
        { path: ["scene", "distractors", "max"], type: "number", label: "Distractors, max", help: "Upper bound. The actual count is sampled in this range." },
        { path: ["scene", "target_max_size_m"], type: "range", label: "Target size, m", help: "The target object is scaled so its longest side lands in this range." },
        { path: ["scene", "distractor_max_size_m"], type: "range", label: "Distractor size, m", help: "Same scaling rule for the other objects." },
        { path: ["scene", "minimum_object_gap_m"], type: "number", label: "Gap between objects, m", help: "Minimum separation so objects do not spawn inside each other." },
        { path: ["scene", "spawn_clearance_m"], type: "number", label: "Spawn clearance, m", help: "How fully an object must fit inside the spawn area." },
        { path: ["scene", "robot_exclusion_half_width_m"], type: "number", label: "Robot exclusion half-width, m", help: "No objects spawn this far left/right of the robot base." },
        { path: ["scene", "robot_exclusion_depth_m"], type: "number", label: "Robot exclusion depth, m", help: "No objects spawn this far from the near table edge toward the robot." },
        { path: ["scene", "settle_seconds"], type: "number", label: "Settle time, s", help: "Physics time after spawn so objects come to rest." },
        { path: ["scene", "timestep_seconds"], type: "number", label: "Physics step, s", help: "Simulator dt. Smaller is more stable and slower." },
      ],
    },
    {
      id: "table",
      title: "Table",
      fields: [
        { path: ["scene", "table", "width_m"], type: "range", label: "Width, m", help: "Random tabletop width along X." },
        { path: ["scene", "table", "depth_m"], type: "range", label: "Depth, m", help: "Random tabletop depth along Y." },
        { path: ["scene", "table", "thickness_m"], type: "number", label: "Thickness, m", help: "Table slab thickness." },
        { path: ["scene", "table", "near_edge_y_m"], type: "number", label: "Near edge Y, m", help: "Table edge on the robot side." },
        { path: ["scene", "table", "spawn_area_size_m"], type: "vec2", label: "Spawn area, m", help: "Rectangle where objects are placed." },
        { path: ["scene", "table", "spawn_area_center_xy_m"], type: "vec2", label: "Spawn area center XY, m", help: "Spawn-area offset in world coordinates." },
        { path: ["scene", "table", "edge_margin_m"], type: "number", label: "Edge margin, m", help: "Objects stay at least this far from the table rim." },
        { path: ["scene", "table", "texture"], type: "string", label: "Table texture", help: "File name in the table textures folder. Empty picks a random texture." },
      ],
    },
    {
      id: "tray",
      title: "Tray",
      fields: [
        { path: ["scene", "tray", "center_xy_m"], type: "vec2", label: "Center XY, m", help: "Tray center on the table. If this key is removed from config, a random non-overlapping spot is used." },
        { path: ["scene", "tray", "inner_width_m"], type: "range", label: "Inner width, m", help: "Random inner width of the tray cavity." },
        { path: ["scene", "tray", "inner_depth_m"], type: "range", label: "Inner depth, m", help: "Random inner depth of the tray cavity." },
        { path: ["scene", "tray", "floor_thickness_m"], type: "range", label: "Floor thickness, m", help: "Tray floor thickness." },
        { path: ["scene", "tray", "wall_thickness_m"], type: "range", label: "Wall thickness, m", help: "Tray wall thickness." },
        { path: ["scene", "tray", "wall_height_m"], type: "range", label: "Wall height, m", help: "How far the walls rise above the floor." },
      ],
    },
    {
      id: "rooms",
      title: "Room",
      fields: [
        { path: ["scene", "rooms", "enabled"], type: "bool", label: "Use a room", help: "If off, the scene keeps an empty background with no skybox." },
        { path: ["scene", "rooms", "name"], type: "string", label: "Room", help: "Skybox folder name under assets/rooms. Empty picks a random prepared room." },
        { path: ["scene", "rooms", "skybox_face_size_px"], type: "number", label: "Skybox face size, px", help: "Resolution of each cube-map face." },
      ],
    },
    {
      id: "appearance",
      title: "Appearance",
      fields: [
        { path: ["scene", "appearance", "table_texture_repeat"], type: "range", label: "Table texture repeat", help: "How finely the tabletop texture is tiled." },
        { path: ["scene", "appearance", "table_reflectance"], type: "range", label: "Table reflectance", help: "Mirror reflection amount written into the table material." },
        { path: ["scene", "appearance", "table_specular"], type: "range", label: "Table specular", help: "Highlight strength of the table material." },
        { path: ["scene", "appearance", "table_shininess"], type: "range", label: "Table shininess", help: "Highlight sharpness of the table material." },
        { path: ["scene", "appearance", "tray_rgb", "min"], type: "vec3", label: "Tray color, min RGB", help: "Lower bound of the random tray color, 0–1." },
        { path: ["scene", "appearance", "tray_rgb", "max"], type: "vec3", label: "Tray color, max RGB", help: "Upper bound of the random tray color, 0–1." },
      ],
    },
    {
      id: "physics",
      title: "Physics",
      fields: [
        { path: ["scene", "physics", "relative_variation"], type: "number", label: "Parameter jitter", help: "Relative noise applied to mass, friction, and other nominals." },
        { path: ["scene", "physics", "gravity_m_s2"], type: "number", label: "Gravity, m/s²", help: "Gravity magnitude." },
        { path: ["scene", "physics", "object_density_kg_m3"], type: "number", label: "Object density, kg/m³", help: "Nominal density when measured mass is missing." },
        { path: ["scene", "physics", "bounding_box_fill_fraction"], type: "number", label: "Bounding-box fill", help: "Fraction of the bounding box treated as volume for mass estimates." },
        { path: ["scene", "physics", "object_mass_kg"], type: "range", label: "Object mass, kg", help: "Clamp on the estimated mass." },
        { path: ["scene", "physics", "object_friction"], type: "vec3", label: "Object friction", help: "MuJoCo sliding / torsional / rolling." },
        { path: ["scene", "physics", "table_friction"], type: "vec3", label: "Table friction", help: "The same three coefficients for the tabletop." },
        { path: ["scene", "physics", "contact_solref"], type: "vec2", label: "Contact solref", help: "Contact stiffness and damping." },
        { path: ["scene", "physics", "contact_solimp"], type: "vec3", label: "Contact solimp", help: "MuJoCo contact impedance." },
      ],
    },
    {
      id: "lights",
      title: "Lights",
      fields: [
        { path: ["scene", "lights", "count"], type: "range", label: "Light count", help: "How many lights to sample. Ignored when the fixed-position list below is not empty." },
        { path: ["scene", "lights", "positions_m"], type: "vec3_list", label: "Fixed positions, m", help: "If this list is not empty, one light is placed at each XYZ and Light count is ignored. Clear the list to sample random positions." },
        { path: ["scene", "lights", "color_temperature_k"], type: "range", label: "Color temperature, K", help: "Light color temperature." },
        { path: ["scene", "lights", "intensity"], type: "range", label: "Intensity", help: "Per-light brightness before the global scale." },
        { path: ["scene", "lights", "total_intensity_scale"], type: "number", label: "Global intensity scale", help: "Multiplies every light." },
        { path: ["scene", "lights", "ambient"], type: "range", label: "Ambient", help: "Ambient share of each light." },
        { path: ["scene", "lights", "specular"], type: "range", label: "Specular", help: "Specular share of each light." },
        { path: ["scene", "lights", "position_x_m"], type: "range", label: "Position X, m", help: "Light spawn range along X." },
        { path: ["scene", "lights", "position_y_m"], type: "range", label: "Position Y, m", help: "Light spawn range along Y." },
        { path: ["scene", "lights", "position_z_m"], type: "range", label: "Position Z, m", help: "Light height above the table." },
        { path: ["scene", "lights", "directional_probability"], type: "number", label: "Directional probability", help: "Chance a light is directional instead of point." },
        { path: ["scene", "lights", "cast_shadow_probability"], type: "number", label: "Shadow probability", help: "Chance a light casts shadows." },
        { path: ["scene", "lights", "attenuation_linear"], type: "range", label: "Linear attenuation", help: "Linear attenuation term for point lights." },
        { path: ["scene", "lights", "attenuation_quadratic"], type: "range", label: "Quadratic attenuation", help: "Quadratic attenuation term." },
        { path: ["scene", "lights", "cutoff_deg"], type: "range", label: "Cutoff, deg", help: "Spot cone angle." },
        { path: ["scene", "lights", "exponent"], type: "range", label: "Exponent", help: "Softness of the spot edge." },
        { path: ["scene", "lights", "headlight_ambient"], type: "range", label: "Headlight ambient", help: "MuJoCo camera headlight ambient." },
        { path: ["scene", "lights", "headlight_diffuse"], type: "range", label: "Headlight diffuse", help: "MuJoCo camera headlight diffuse." },
      ],
    },
  ],
  cameras: [
    {
      id: "setup",
      title: "Set",
      fields: [
        { path: ["cameras", "count"], type: "range", label: "Camera count", help: "How many views a scene gets, including the required ones. Extra views are sampled from unused mounts." },
        { path: ["cameras", "required"], type: "string_list", label: "Required cameras", help: "These cameras are always present. The rest are sampled from mounts." },
        { path: ["compute", "policy_map"], type: "policy_map", label: "Policy camera map", help: "Shared slot map for SFT, GRPO, and Inference. camera1 is the first model view, camera2 the second. Empty leaves the slot unused. Names must exist on the scene. SmolVLA uses up to 5 filled slots. ACT uses the first N checkpoint slots." },
        { path: ["cameras", "fast_pipeline"], type: "bool", label: "Fast pipeline", help: "Lighter post-process (noise, exposure) without the heavy optics path." },
        { path: ["cameras", "frame_rate_hz"], type: "range", label: "Frame rate, Hz", help: "Shared FPS for every camera in the scene. Affects exposure and blur." },
      ],
    },
    {
      id: "optics",
      title: "Optics",
      fields: [
        { path: ["cameras", "fovy_clamp_deg"], type: "range", label: "Global FOV clamp, deg", help: "Hard limits on vertical FOV after it is computed from the sensor." },
        { path: ["cameras", "fovy_limits_by_camera", "front"], type: "range", label: "Front FOV, deg", help: "Allowed FOV for the front camera." },
        { path: ["cameras", "fovy_limits_by_camera", "overview"], type: "range", label: "Overview FOV, deg", help: "Allowed FOV for the overview camera." },
        { path: ["cameras", "fovy_limits_by_camera", "left"], type: "range", label: "Left FOV, deg", help: "Allowed FOV for the left camera." },
        { path: ["cameras", "fovy_limits_by_camera", "right"], type: "range", label: "Right FOV, deg", help: "Allowed FOV for the right camera." },
        { path: ["cameras", "fovy_limits_by_camera", "top"], type: "range", label: "Top FOV, deg", help: "Allowed FOV for the top camera." },
        { path: ["cameras", "fovy_limits_by_camera", "wrist"], type: "range", label: "Wrist FOV, deg", help: "Allowed FOV for the wrist camera." },
      ],
    },
    {
      id: "placement",
      title: "Placement",
      fields: [
        { path: ["cameras", "position_jitter_m"], type: "number", label: "Position jitter, m", help: "Random offset of external cameras around their mounts." },
        { path: ["cameras", "look_at_m"], type: "vec3", label: "Look-at point, m", help: "World point the external cameras look at." },
        { path: ["cameras", "look_at_jitter_m"], type: "vec3", label: "Look-at jitter, m", help: "Noise on the look-at point along XYZ." },
        { path: ["cameras", "roll_jitter_deg"], type: "number", label: "Roll jitter, deg", help: "Random roll around the view axis." },
        { path: ["cameras", "wrist_position_jitter_m"], type: "number", label: "Wrist position jitter, m", help: "Offset of the wrist camera relative to the robot model." },
        { path: ["cameras", "wrist_rotation_jitter_deg"], type: "number", label: "Wrist rotation jitter, deg", help: "Random rotation of the wrist camera." },
        { path: ["cameras", "mounts", "front", "position_m"], type: "vec3", label: "Front mount, m", help: "Base position of the front camera." },
        { path: ["cameras", "mounts", "overview", "position_m"], type: "vec3", label: "Overview mount, m", help: "Base position of the overview camera." },
        { path: ["cameras", "mounts", "left", "position_m"], type: "vec3", label: "Left mount, m", help: "Base position of the left camera." },
        { path: ["cameras", "mounts", "right", "position_m"], type: "vec3", label: "Right mount, m", help: "Base position of the right camera." },
        { path: ["cameras", "mounts", "top", "position_m"], type: "vec3", label: "Top mount, m", help: "Base position of the top camera." },
      ],
    },
    {
      id: "profiles",
      title: "Sensor profiles",
      fields: [
        { path: ["cameras", "profiles", "budget", "probability"], type: "number", label: "Budget mix", help: "Chance a camera samples the cheap-webcam profile. Weights are normalized across the three profiles." },
        { path: ["cameras", "profiles", "midrange", "probability"], type: "number", label: "Midrange mix", help: "Chance a camera samples the midrange profile. Preset 1, the other two 0." },
        { path: ["cameras", "profiles", "normal", "probability"], type: "number", label: "Normal mix", help: "Chance a camera samples the cleaner profile." },
      ],
    },
  ],
  language: [
    {
      id: "prompts",
      title: "Prompts",
      fields: [
        { path: ["language", "prompts", "templates"], type: "string_list", label: "Instruction templates", help: "One template is sampled at random. Placeholders: {target}, {destination}." },
        { path: ["language", "prompts", "destination_templates"], type: "string_list", label: "Destination templates", help: "How the tray is named in the instruction. Placeholder: {tray_color}." },
      ],
    },
  ],
  rewards: [
    {
      id: "impulse",
      title: "Impulse",
      fields: [
        { path: ["reward", "impulse", "max_penalty"], type: "number", label: "Max penalty", help: "Cap on hit/shove penalty. Preset 0.25: violence is costly, a clean success still wins." },
        { path: ["reward", "impulse", "normalization_percentile"], type: "number", label: "Normalization percentile", help: "Percentile used to normalize impulse." },
        { path: ["reward", "impulse", "include_target_to_distractor"], type: "bool", label: "Penalize target–distractor hits", help: "Apply a penalty if the target hits another object." },
      ],
    },
    {
      id: "contact",
      title: "Contact and proximity",
      fields: [
        { path: ["reward", "target_contact", "max_seconds"], type: "number", label: "Contact, max seconds", help: "Seconds of pad contact for the full bonus. Preset 2." },
        { path: ["reward", "target_contact", "max_reward"], type: "number", label: "Contact, max reward", help: "Small stage bonus. Preset 0.15 so touching cannot beat a grasp." },
        { path: ["reward", "target_proximity", "decay_distance_m"], type: "number", label: "Decay distance, m", help: "TCP–target distance where proximity is ~e⁻¹. Preset 0.15 m." },
        { path: ["reward", "target_proximity", "max_seconds"], type: "number", label: "Proximity, max seconds", help: "Window for accumulating proximity. Preset 20." },
        { path: ["reward", "target_proximity", "max_reward"], type: "number", label: "Proximity, max reward", help: "Small approach bonus. Preset 0.1 so hovering does not farm reward." },
      ],
    },
    {
      id: "grasp",
      title: "Progress and grasp",
      fields: [
        { path: ["reward", "progress", "max_reward"], type: "number", label: "Progress, max reward", help: "Best move of the target toward the tray. Preset 0.25." },
        { path: ["reward", "grasp", "required_contact_seconds"], type: "number", label: "Seconds to count a grasp", help: "Both pads on the target. Preset 0.35 so a brush does not count." },
        { path: ["reward", "grasp", "loss_reset_seconds"], type: "number", label: "Reset after loss, s", help: "Pause after losing a grasp before a new one can count. Preset 0.4." },
        { path: ["reward", "grasp", "reward_per_grasp"], type: "number", label: "Reward per grasp", help: "One real grasp. Preset 0.3." },
        { path: ["reward", "grasp", "max_reward"], type: "number", label: "Grasp, max reward", help: "Cap so grasp-farming does not pay. Preset 0.3." },
      ],
    },
    {
      id: "success",
      title: "Success",
      fields: [
        { path: ["reward", "success", "base_reward"], type: "number", label: "Base reward", help: "Must dominate shaping. Preset 2." },
        { path: ["reward", "success", "max_speed_bonus"], type: "number", label: "Speed bonus", help: "Keep tiny. Preset 0.1 so GRPO does not learn to throw." },
        { path: ["reward", "success", "stable_seconds"], type: "number", label: "Stable time, s", help: "How long the object must rest in the tray. Preset 0.8." },
        { path: ["reward", "success", "max_linear_speed_m_s"], type: "number", label: "Max linear speed, m/s", help: "At-rest threshold. Preset 0.04." },
      ],
    },
  ],
  compute: [
    {
      id: "render",
      title: "Render",
      fields: [
        { path: ["compute", "render"], type: "device", label: "Render device", help: "Auto picks a GPU with a display engine. Compute-only cards (Tesla, P100, …) are not used for render." },
        { path: ["compute", "model"], type: "model_device", label: "Model device", help: "Where the custom SmolVLA loop runs. Auto uses the first CUDA GPU." },
      ],
    },
  ],
  grpo: [
    {
      id: "loop",
      title: "Online loop",
      fields: [
        { path: ["environment", "grpo", "train_scope"], type: "select", label: "Train scope", help: "π_RL / Flow-SDE default is the action expert only. Full VLA also updates the VLM.", options: [{ value: "experts", label: "Action expert only" }, { value: "full_vla", label: "Full VLA" }] },
        { path: ["environment", "grpo", "group_size"], type: "number", label: "Group size", help: "Rollouts of the same scene used for group-relative advantage." },
        { path: ["environment", "grpo", "parallel_rollouts"], type: "number", label: "Together", help: "How many group members are inferred at once. Capped by group size. Same as Together on the GRPO page." },
        { path: ["environment", "grpo", "scenes_per_update"], type: "number", label: "Scenes per update", help: "How many scenes are rolled out before one GRPO step. Preset: 16 (256 trajectories with group 16)." },
        { path: ["environment", "grpo", "scene_waves"], type: "number", label: "Scene waves", help: "How many of those scenes roll at once. Capped by scenes per update. Together is the infer batch; a free scene wave steps physics." },
        { path: ["environment", "grpo", "max_updates"], type: "number", label: "Max updates", help: "Stop after this many optimizer steps. Empty or 0 means run until Stop." },
        { path: ["environment", "grpo", "n_action_steps"], type: "number", label: "Actions per chunk", help: "How many actions from each flow chunk are executed in the env." },
        { path: ["environment", "grpo", "clip_eps"], type: "number", label: "PPO clip ε", help: "Trust region vs the rollout-time policy. Typical: 0.2." },
        { path: ["environment", "grpo", "kl_coef"], type: "number", label: "KL coefficient", help: "Anchor to the frozen SFT policy (k3). Flow-GRPO uses this so RL cannot walk off the SFT init. Preset 0.18." },
      ],
    },
    {
      id: "flow",
      title: "Flow-SDE",
      fields: [
        { path: ["environment", "grpo", "flow", "sde_mode"], type: "select", label: "SDE mode", help: "one_random_step is Flow-GRPO-Fast. all_steps is full Flow-SDE on every denoising step.", options: [{ value: "one_random_step", label: "One random SDE step" }, { value: "all_steps", label: "All SDE steps" }] },
        { path: ["environment", "grpo", "flow", "denoising_steps"], type: "number", label: "Denoising steps", help: "Flow steps during GRPO rollouts. Same N as inference unless you change it." },
        { path: ["environment", "grpo", "flow", "noise_level"], type: "number", label: "SDE noise a", help: "σ_τ = a · √(τ/(1-τ)). RLinf uses 0.5. Preset 0.35 is the SmolVLA_RL value for a weaker SFT start." },
        { path: ["environment", "grpo", "flow", "flow_scale", "enabled"], type: "bool", label: "Flow-scale weights", help: "Weight early (noisier) τ more when picking the SDE step or averaging the GRPO loss." },
        { path: ["environment", "grpo", "flow", "flow_scale", "power"], type: "number", label: "Scale power", help: "τ^power. 0.5 favors high-noise steps without ignoring the rest." },
        { path: ["environment", "grpo", "flow", "flow_scale", "uniform_mix"], type: "number", label: "Uniform mix", help: "0 is pure τ^power. 1 is uniform over denoising steps." },
        { path: ["environment", "grpo", "flow", "flow_scale", "min_weight"], type: "number", label: "Min weight", help: "Lower bound after rescaling the flow-scale curve." },
        { path: ["environment", "grpo", "flow", "flow_scale", "max_weight"], type: "number", label: "Max weight", help: "Upper bound after rescaling the flow-scale curve." },
      ],
    },
    {
      id: "optim",
      title: "Optimizer",
      fields: [
        { path: ["environment", "grpo", "optimization", "action_expert_learning_rate"], type: "number", label: "Expert LR", help: "AdamW LR for the action expert. Preset: 5e-6." },
        { path: ["environment", "grpo", "optimization", "vlm_learning_rate"], type: "number", label: "VLM LR", help: "Used only when train scope is Full VLA. Typical: 1e-6." },
        { path: ["environment", "grpo", "optimization", "weight_decay"], type: "number", label: "Weight decay", help: "AdamW weight decay. RLinf openpi GRPO uses 0.01." },
        { path: ["environment", "grpo", "optimization", "adam_beta1"], type: "number", label: "AdamW β1", help: "First moment." },
        { path: ["environment", "grpo", "optimization", "adam_beta2"], type: "number", label: "AdamW β2", help: "Second moment." },
        { path: ["environment", "grpo", "optimization", "adam_eps"], type: "number", label: "AdamW epsilon", help: "Numerical stability term." },
        { path: ["environment", "grpo", "optimization", "update_microbatch_expert_only"], type: "number", label: "Microbatch, expert", help: "Chunks whose log-probs are recomputed together when the VLM is frozen." },
        { path: ["environment", "grpo", "optimization", "update_microbatch_full_vla"], type: "number", label: "Microbatch, full VLA", help: "Same, when the VLM is trainable. Keep this small." },
        { path: ["environment", "grpo", "optimization", "max_grad_norm"], type: "number", label: "Grad clip", help: "Global grad-norm clip before the optimizer step." },
      ],
    },
    {
      id: "runtime",
      title: "Checkpointing and smoke",
      fields: [
        { path: ["environment", "grpo", "checkpointing", "save_every_scenes"], type: "number", label: "Save every N scenes", help: "Write a LeRobot checkpoint after this many generated scenes." },
        { path: ["environment", "grpo", "checkpointing", "keep_last"], type: "number", label: "Keep last", help: "How many numbered checkpoints to retain." },
        { path: ["environment", "grpo", "checkpointing", "keep_optimizer"], type: "bool", label: "Save optimizer", help: "Store AdamW moments next to the policy." },
        { path: ["environment", "grpo", "visualization", "enabled"], type: "bool", label: "Live rollouts", help: "Stream cameras of the current group member on the GRPO page." },
        { path: ["environment", "grpo", "visualization", "stream_fps"], type: "number", label: "Stream FPS", help: "MJPEG rate for the GRPO camera tiles." },
        { path: ["environment", "grpo", "smoke", "enabled"], type: "bool", label: "Smoke overrides", help: "If yes, the next run uses the short smoke group/scene/duration below." },
        { path: ["environment", "grpo", "smoke", "scenes_per_update"], type: "number", label: "Smoke scenes / update", help: "Used only when smoke overrides are on." },
        { path: ["environment", "grpo", "smoke", "group_size"], type: "number", label: "Smoke group size", help: "Used only when smoke overrides are on." },
        { path: ["environment", "grpo", "smoke", "duration_seconds"], type: "number", label: "Smoke duration, s", help: "Used only when smoke overrides are on." },
      ],
    },
  ],
  training: [
    {
      id: "optimization",
      title: "Optimizer",
      fields: [
        { path: ["training", "optimizer_beta1"], type: "number", label: "AdamW β1", help: "Used by SmolVLA and ACT. Preset: 0.9." },
        { path: ["training", "optimizer_beta2"], type: "number", label: "AdamW β2", help: "Used by SmolVLA and ACT. Preset: 0.95." },
        { path: ["training", "optimizer_eps"], type: "number", label: "AdamW epsilon", help: "Used by SmolVLA and ACT. Preset: 1e-8." },
        { path: ["training", "optimizer_weight_decay"], type: "number", label: "Weight decay", help: "Used by SmolVLA and ACT. SmolVLA preset: 1e-10. ACT is fine with 1e-4." },
        { path: ["training", "grad_clip_norm"], type: "number", label: "Gradient clip norm", help: "Used by SmolVLA and ACT. Preset: 10. Set 0 to disable clipping." },
      ],
    },
    {
      id: "scheduler",
      title: "Cosine scheduler",
      fields: [
        { path: ["training", "scheduler_warmup_steps"], type: "number", label: "Warmup steps", help: "Linear rise to peak. Preset: 1000." },
        { path: ["training", "scheduler_decay_steps"], type: "number", label: "Decay steps", help: "ACT: hold peak, then cosine-decay this many steps at the end. SmolVLA SFT ignores this and decays over the whole run." },
        { path: ["training", "scheduler_decay_lr"], type: "number", label: "Final learning rate", help: "Used by SmolVLA and ACT. Cosine floor. Preset: 2.5e-6." },
      ],
    },
    {
      id: "runtime",
      title: "Runtime",
      fields: [
        { path: ["training", "num_workers"], type: "number", label: "Data workers", help: "SmolVLA video decode workers. ACT loads frames into RAM once, so this is unused there." },
        { path: ["training", "vision_cache"], type: "bool", label: "Vision cache", help: "SmolVLA only. Reuse frozen SigLIP embeddings on CPU." },
        { path: ["training", "vision_cache_gb"], type: "number", label: "Vision cache, GB", help: "SmolVLA only. CPU cap for cached embeddings. 0 turns the cache off." },
        { path: ["training", "frame_cache"], type: "bool", label: "Frame cache", help: "SmolVLA decoded-frame cache. ACT always caches resized frames in RAM." },
        { path: ["training", "frame_cache_gb"], type: "number", label: "Frame cache, GB", help: "SmolVLA only. Total CPU cap for all workers together." },
        { path: ["training", "use_amp"], type: "bool", label: "Mixed precision", help: "Used by SmolVLA and ACT. Faster and less VRAM when the model GPU supports it." },
      ],
    },
  ],
  rollout: [
    {
      id: "episode",
      title: "Episode",
      fields: [
        { path: ["environment", "rollout", "duration_seconds"], type: "number", label: "Duration, s", help: "Simulated episode length. Steps = duration × control Hz. Same value as Time on GRPO / Inference / Eval unless that page overrides it for one run." },
        { path: ["environment", "rollout", "control_hz"], type: "number", label: "Control Hz", help: "How often the policy and the simulator take a step. Also written into new scenes." },
        { path: ["environment", "rollout", "actions_are_degrees"], type: "bool", label: "Actions are degrees", help: "If yes, env actions are converted from degrees to radians. SO-ARM datasets use degrees." },
      ],
    },
    {
      id: "image",
      title: "Image",
      fields: [
        { path: ["environment", "rollout", "camera_width"], type: "number", label: "Camera width, px", help: "Render width for policy cameras and scene offscreen buffers." },
        { path: ["environment", "rollout", "camera_height"], type: "number", label: "Camera height, px", help: "Render height for policy cameras and scene offscreen buffers." },
      ],
    },
  ],
  eval: [
    {
      id: "valset",
      title: "Valset",
      fields: [
        { path: ["environment", "eval", "size"], type: "number", label: "Size", help: "How many validation scenes to keep. Same as Size on the Eval page." },
        { path: ["environment", "eval", "seed"], type: "number", label: "Seed", help: "First scene seed. Following scenes are seed+1…" },
      ],
    },
    {
      id: "run",
      title: "Run",
      fields: [
        { path: ["environment", "eval", "duration_seconds"], type: "number", label: "Duration, s", help: "Max simulated length of one eval scene." },
        { path: ["environment", "eval", "parallel"], type: "number", label: "Together", help: "How many val scenes run at once." },
        { path: ["environment", "eval", "denoising_steps"], type: "number", label: "Denoising steps", help: "ODE flow steps during eval. Same meaning as Denoise on the Eval page." },
        { path: ["environment", "eval", "n_action_steps"], type: "number", label: "Actions per chunk", help: "How many actions from each infer chunk are executed." },
      ],
    },
  ],
};

export const SCENE_KEYS = new Set(["seed", "scene", "cameras", "language", "compute"]);
