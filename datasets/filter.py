class GlobalCategoryFilter:
    def get_filter():
        door_window = ["window", "door"]

        second_tier = [
            "table_lamp",
            "chandelier",
            "guitar",
            "amplifier",
            "keyboard",
            "drumset",
            "microphone",
            "accordion",
            "toy",
            "xbox",
            "playstation",
            "fishbowl",
            "chessboard",
            "iron",
            "helmet",
            "telephone",
            "stationary_container",
            "ceiling_fan",
            "bottle",
            "fruit_bowl",
            "glass",
            "knife_rack",
            "plates",
            "books",
            "book",
            "television",
            "wood_board",
            "switch",
            "pillow",
            "laptop",
            "clock",
            "helmet",
            "bottle",
            "trinket",
            "glass",
            "range_hood",
            "candle",
            "soap_dish",
        ]

        wall_objects = ["wall_lamp", "mirror", "curtain", "blind"]

        unimportant = [
            "toy",
            "fish_tank",
            "tricycle",
            "vacuum_cleaner",
            "weight_scale",
            "heater",
            "picture_frame",
            "beer",
            "shoes",
            "weight_scale",
            "decoration",
            "ladder",
            "tripod",
            "air_conditioner",
            "cart",
            "fireplace_tools",
            "vase",
        ]

        inhabitants = ["person", "cat", "bird", "dog", "pet"]

        special_filter = [
            "rug",
        ]

        filtered = (
            second_tier + unimportant + inhabitants + special_filter + wall_objects
        )

        unwanted_complex_structure = ["partition", "column", "arch", "stairs"]
        set_items = [
            "chair_set",
            "stereo_set",
            "table_and_chair",
            "slot_machine_and_chair",
            "kitchen_set",
            "double_desk",
            "double_desk_with_chairs",
            "dressing_table_with_stool",
            "kitchen_island_with_range_hood_and_table",
        ]

        outdoor = [
            "lawn_mower",
            "car",
            "motorcycle",
            "bicycle",
            "garage_door",
            "outdoor_seating",
            "fence",
        ]

        rejected = unwanted_complex_structure + set_items + outdoor

        return filtered, rejected, door_window

    def get_bedroom_filter():
        bedroom_filter = [
            "double_bed",
            "desk",
            "office_chair",
            "computer",
            "wardrobe_cabinet",
            "stand",
            "tv_stand",
            "chair",
            "single_bed",
            "dressing_table",
            "ottoman",
            "coffee_table",
            "armchair",
            "bunker_bed",
            "plant",
            "dresser",
            "baby_bed",
            "floor_lamp",
            "pedestal_fan",
            "ironing_board",
            "sofa",
            "hanger",
            "shelving",
            "shoes_cabinet",
            "workplace",
            "loudspeaker",
            "bookshelf",
            "roof",
        ]
        return bedroom_filter
