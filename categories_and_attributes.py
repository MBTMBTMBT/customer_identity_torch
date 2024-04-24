class CategoriesAndAttributes:
    mask_categories: list[str] = []
    merged_categories: dict[str, list[str]] = {}
    mask_labels: list[str] = []
    selective_attributes: dict[str, list[str]] = {}
    plane_attributes: list[str] = []
    avoided_attributes: list[str] = []
    attributes: list[str] = []
    thresholds_mask: dict[str, float] = {}
    thresholds_pred: dict[str, float] = {}


class CelebAMaskHQCategoriesAndAttributes(CategoriesAndAttributes):
    mask_categories = ['cloth', 'r_ear', 'hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow',
                       'r_ear', 'r_eye', 'skin', 'u_lip', 'hat', 'l_ear', 'neck_l', 'eye_g', ]
    merged_categories = {
        'ear': ['l_ear', 'r_ear', ],
        'brow': ['l_brow', 'r_brow', ],
        'eye': ['l_eye', 'r_eye', ],
        'mouth': ['l_lip', 'u_lip', 'mouth', ],
    }
    _categories_to_merge = []
    for key in sorted(list(merged_categories.keys())):
        for cat in merged_categories[key]:
            _categories_to_merge.append(cat)
    for key in mask_categories:
        if key not in _categories_to_merge:
            merged_categories[key] = [key]
    mask_labels = ['hair']
    selective_attributes = {
        'facial_hair': ['5_o_Clock_Shadow', 'Goatee', 'Mustache', 'No_Beard', 'Sideburns', ],
        'hair_colour': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', ],
        'hair_shape': ['Straight_Hair', 'Wavy_Hair', ]
    }
    plane_attributes = ['Bangs', 'Eyeglasses', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Necklace',
                        'Wearing_Necktie', 'Male', ]
    avoided_attributes = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby',
                          'Double_Chin', 'High_Cheekbones', 'Narrow_Eyes', 'Oval_Face', 'Pointy_Nose',
                          'Receding_Hairline', 'Rosy_Cheeks', 'Heavy_Makeup', 'Wearing_Lipstick', 'Attractive',
                          'Blurry', 'Mouth_Slightly_Open', 'Pale_Skin', 'Smiling', 'Young', ]
    attributes = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
                  "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
                  "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                  "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                  "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
                  "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

    thresholds_mask: dict[str, float] = {}
    thresholds_pred: dict[str, float] = {}

    # set default thresholds:
    for key in sorted(merged_categories.keys()):
        thresholds_mask[key] = 0.5
    for key in attributes + mask_labels:
        thresholds_pred[key] = 0.5

    # set specific thresholds:
    thresholds_mask['eye_g'] = 0.25
    thresholds_pred['Eyeglasses'] = 0.25
    thresholds_pred['Wearing_Earrings'] = 0.5
    thresholds_pred['Wearing_Necklace'] = 0.5
    thresholds_pred['Wearing_Necktie'] = 0.5


class CCPCategoriesAndAttributes(CategoriesAndAttributes):
    mask_categories = [
        'null', 'accessories', 'bag', 'belt', 'blazer', 'blouse', 'bodysuit', 'boots', 'bra', 'bracelet', 'cape',
        'cardigan', 'clogs', 'coat', 'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair', 'hat', 'heels',
        'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings', 'loafers', 'necklace', 'panties', 'pants',
        'pumps', 'purse', 'ring', 'romper', 'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers',
        'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear', 't-shirt', 'tie', 'tights',
        'top', 'vest', 'wallet', 'watch', 'wedges',
    ]
    merged_categories = {
        'legwear': ['leggings', 'tights', 'stockings', ],
        'footwear': ['shoes', 'pumps', 'flats', 'wedges', 'heels', 'clogs', 'sneakers', 'sandals', 'boots',
                     'loafers', ],
        'outwear': ['jacket', 'coat', 'blazer', 'cape', 'suit', 'cardigan', 'hoodie', ],
        'glasses': ['glasses', 'sunglasses', ],
        'tops': ['shirt', 'sweater', 'cardigan', 'vest', 'sweatshirt', 'blouse', 't-shirt', 'top', 'bodysuit',
                 'jumper', ],
        'bottoms': ['pants', 'skirt', 'shorts', 'jeans', ],
        'accessories': ['bracelet', 'purse', 'wallet', 'accessories', 'ring', 'watch', 'bag', ],
    }
    _categories_to_merge = []
    for key in sorted(list(merged_categories.keys())):
        for cat in merged_categories[key]:
            _categories_to_merge.append(cat)
    for key in mask_categories:
        if key not in _categories_to_merge:
            merged_categories[key] = [key]
    mask_labels = []
    selective_attributes = {
        'footwear': ['shoes', 'pumps', 'flats', 'wedges', 'heels', 'clogs', 'sneakers', 'sandals', 'boots',
                     'loafers', ],
        'outwear': ['jacket', 'coat', 'blazer', 'cape', 'suit', 'cardigan', 'hoodie', ],
        'glasses': ['glasses', 'sunglasses', ],
        'tops': ['shirt', 'sweater', 'cardigan', 'vest', 'sweatshirt', 'blouse', 't-shirt', 'top', 'bodysuit',
                 'jumper', ],
        'trousers': ['pants', 'shorts', 'jeans', ],
        'dress': ['dress, skirt', ]
    }
    plane_attributes = []
    avoided_attributes = ['bra', ]
    attributes = mask_categories

    thresholds_mask: dict[str, float] = {}
    thresholds_pred: dict[str, float] = {}

    # set default thresholds:
    for key in sorted(merged_categories.keys()):
        thresholds_mask[key] = 0.5
    for key in attributes + mask_labels:
        thresholds_pred[key] = 0.5

    # set specific thresholds:
    # examples:
    thresholds_mask['glasses'] = 0.5
    thresholds_pred['glasses'] = 0.5
