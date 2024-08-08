
class BaseConfig:
    expreriment    = "default"
    robot          = "default"
    mjcf_file_path = ""
    decimation     = 2,
    sync           = True,
    headless       = False,
    mirror_image   = False
    render_set     = {
        "fps"    : 24,
        "width"  : 1280,
        "height" :  720
    }
    put_text       = True
    obs_camera_id  = -1
    rb_link_list   = []
    obj_list       = []
