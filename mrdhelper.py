# MRD Helper functions
import ismrmrd
import re

def update_img_header_from_raw(imgHead, rawHead):
    if rawHead is None:
        return imgHead

    imgHead.version                = rawHead.version
    imgHead.flags                  = rawHead.flags
    imgHead.measurement_uid        = rawHead.measurement_uid

    # # These fields are not translated from the raw header, but filled in
    # # during image creation by from_array
    # imgHead.data_type            = 
    # imgHead.matrix_size          = 
    # imgHead.field_of_view        = 

    imgHead.image_type             = ismrmrd.IMTYPE_MAGNITUDE

    imgHead.channels               = 1
    imgHead.position               = rawHead.position
    imgHead.read_dir               = rawHead.read_dir
    imgHead.phase_dir              = rawHead.phase_dir
    imgHead.slice_dir              = rawHead.slice_dir
    imgHead.patient_table_position = rawHead.patient_table_position

    imgHead.average                = rawHead.idx.average
    imgHead.slice                  = rawHead.idx.slice
    imgHead.contrast               = rawHead.idx.contrast
    imgHead.phase                  = rawHead.idx.phase
    imgHead.repetition             = rawHead.idx.repetition
    imgHead.set                    = rawHead.idx.set

    imgHead.acquisition_time_stamp = rawHead.acquisition_time_stamp
    imgHead.physiology_time_stamp  = rawHead.physiology_time_stamp

    # Defaults, to be updated by the user
    imgHead.image_type             = ismrmrd.IMTYPE_MAGNITUDE
    imgHead.image_index            = 0
    imgHead.image_series_index     = 0

    imgHead.user_float             = rawHead.user_float
    imgHead.user_int               = rawHead.user_int

    return imgHead

def get_meta_value(meta, key):
    # Get a value from MRD Meta Attributes (blank if key not found)
    if key in meta.keys():
        return meta[key]
    else:
        return None

def extract_minihead_bool_param(miniHead, name):
    # Extract a bool parameter from the serialized text of the ICE MiniHeader
    # Note: if missing, return false (following ICE logic)
    expr = r'(?<=<ParamBool."' + name + r'">{)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return False
    else:
        if res.group(0).strip().lower() == '"true"'.lower():
            return True
        else:
            return False

def extract_minihead_long_param(miniHead, name):
    # Extract a long parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamLong."' + name + r'">{)\s*\d*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return 0
    else:
        return int(res.group(0))

def extract_minihead_double_param(miniHead, name):
    # Extract a double parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamDouble."' + name + r'">{)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return float(0)
    else:
        return float(res.group(0))

def extract_minihead_string_param(miniHead, name):
    # Extract a string parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamString."' + name + r'">{)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    else:
        return res.group(0).strip()