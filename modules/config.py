from pydantic import BaseModel


class SegmentCategories(BaseModel):
    background: bool = False
    hat: bool = False
    hair: bool = False
    sunglasses: bool = False
    upper_clothes: bool = False
    skirt: bool = False
    pants: bool = False
    dress: bool = False
    belt: bool = False
    left_shoe: bool = False
    right_shoe: bool = False
    face: bool = False
    left_leg: bool = False
    right_leg: bool = False
    left_arm: bool = False
    right_arm: bool = False
    bag: bool = False
    scarf: bool = False