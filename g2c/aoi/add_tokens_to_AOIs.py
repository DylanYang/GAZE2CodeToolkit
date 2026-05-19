import pandas as pd


def add_tokens_to_AOIs(file_path: str, aois_raw: pd.DataFrame) -> pd.DataFrame:
    """Adds tokens from code files to aois dataframe and returns it.

    Parameters
    ----------
    file_path : str
        path to directory where code files are stored. In EMIP this is "emip_stimulus_programs"

    aois_raw : pandas.Dataframe
        the dataframe where AOIs are stored.

    Returns
    -------
    pandas.DataFrame
        a dataframe of AOIs with token information
    """

    image_name = aois_raw["image"][1]

    # rectangle files
    if image_name == "rectangle_java.jpg":
        file_name = "Rectangle.java"

    if image_name == "rectangle_java2.jpg":
        file_name = "Rectangle.java"

    if image_name == "rectangle_python.jpg":
        file_name = "Rectangle.py"

    if image_name == "rectangle_scala.jpg":
        file_name = "Rectangle.scala"

    # vehicle files 
    if image_name == "vehicle_java.jpg":
        file_name = "Vehicle.java"

    if image_name == "vehicle_java2.jpg":
        file_name = "Vehicle.java"

    if image_name == "vehicle_python.jpg":
        file_name = "vehicle.py"

    if image_name == "vehicle_scala.jpg":
        file_name = "Vehicle.scala"

    code_file = open(file_path + file_name)

    code_text = code_file.read()

    code_line = code_text.replace('\t', '').replace(
        '        ', '').replace('    ', '').split('\n')

    filtered_line = []

    for line in code_line:
        if len(line) != 0:
            filtered_line.append(line.split(' '))

    # after the code file has been tokenized and indexed
    # we can attach tokens to correct AOI

    # aois_raw = aois_raw[aois_raw.kind == "sub-line"].copy()

    # tokens = []

    # for location in aois_raw["name"].iteritems():
    #     line_part = location[1].split(' ')
    #     line_num = int(line_part[1])
    #     part_num = int(line_part[3])

    #     # print(line_part, filtered_line[line_num - 1])
    #     tokens.append(filtered_line[line_num - 1][part_num - 1])

    # aois_raw["token"] = tokens

    # newest panda --> 'Series' object has no attribute 'iteritems'

    # Filter the DataFrame
    aois_raw = aois_raw[aois_raw.kind == "sub-line"].copy()

    tokens = []

    # Iterate over the 'name' Series using items()
    for location in aois_raw["name"].items():
        line_part = location[1].split(' ')
        line_num = int(line_part[1])
        part_num = int(line_part[3])
        tokens.append({
            "line_num": line_num,
            "part_num": part_num
        })

    # Convert tokens list to DataFrame
    tokens_df = pd.DataFrame(tokens)

    aois_raw["token"] = tokens

    if aois_raw[aois_raw['token'] == '']['name'].count() != 0:
        print("Error in adding tokens, some tokens are missing!")

    return aois_raw
