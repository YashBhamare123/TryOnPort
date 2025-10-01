from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    resized_height : int
    resized_width : int = 0
    keep_ratio : bool = True
    resize_mode : str = 'bilinear'
    crop_padding : int = 16
    grow_padding : int = 20