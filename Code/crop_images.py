from random import sample
from PIL import Image
from zipfile import ZipFile
import os, sys

## coordinates for PIL need to be x_min, y_min, x_max, y_max)
coord_dict = {
    "Cornwall_BodminFlaxmoor": (214,328,507,475),
    "Cornwall_BodminPetrocsWell_Screen": (773,539,918,607),
    "Cornwall_BoscastleMarine": (252,216,535,597),
    "Cornwall_BudeCedarGrove": (170,275,411,424),
    "Cornwall_CawsandCP": (301,249,371,309),
    "Cornwall_Crinnis": (340,300,540,420),
    "Cornwall_GorranHaven": (245,215,373,334),
    "Cornwall_IdlessDam_cam1": (520,354,1222,932),
    "Cornwall_KingsandCP": (168,326,429,419),
    "Cornwall_KingsandPP": (119,91,406,209),
    "Cornwall_LostwithielUP_Screen": (571,376,1041,716),
    "Cornwall_LostwithielUP_Upstream": (469,306,686,396),
    "Cornwall_LostwithielWW": (337,210,460,360),
    "Cornwall_Mevagissey_MainScreen": (216,351,351,421),
    "Cornwall_Mevagissey_PreScreen": (427,104,563,183),
    "Cornwall_PenrynTP": (377,225,520,354),
    "Cornwall_PenzanceCC": (256,267,407,340),
    "Cornwall_PenzanceCS": (277,249,488,442),
    "Cornwall_PlymCotHill": (192,206,461,309),
    "Cornwall_PlymptonChaddlewood_MainScreen": (306,718,1058,974),
    "Cornwall_PlymptonForSt": (222,100,482,478),
    "Cornwall_PlymptonKay": (217,260,427,444),
    "Cornwall_PolperroLangreek": (187,312,388,478),
    "Cornwall_PorthlevenScreen": (276,245,417,409),
    "Cornwall_Portreath": (150,285,420,420),
    "Cornwall_StIvesConsols_Screen": (669,475,1060,603),
    "Cornwall_StIvesConsols_Upstream": (609,192,796,637),
    "Cornwall_TamertonFoliot": (233,283,458,370),
    "Cornwall_WadebridgePolmorla": (155,220,344,409),
    "Devon_BarnstapleBradiford": (113,199,325,581),
    "Devon_BarnstapleConeyGut_Screen": (438,338,630,443),
    "Devon_BarnstaplePortmarshLane": (108,145,496,362),
    "Devon_Buckfastleigh": (208,621,1768,1054),
    "Devon_Dulverton": (162,269,595,477),
    "Devon_Holbeam_Impound": (138,355,391,571),
    "Devon_KenwithValleyChannelScreen": (116,184,359,429),
    "Devon_LympstoneScreen": (369,261,617,441),
    "Devon_OtterySMChapelLane": (224,178,468,478),
    "Devon_PalmersDam_Upstream": (829,333,1022,605),
    "Devon_SwimbridgeScreen": (330,139,506,344),
    "sites_ashtoncollitersbrooksplitter_cam2": (237,382,374,575),
    "sites_brutondam_cam1": (43,209,156,291),
    "sites_corsham_cam1": (185,378,325,572),
    "sites_corshamaqueduct_cam1": (1539,654,1824,897),
    "sites_eastville_cam2": (635,133,811,211),
    "sites_midsomernorton_cam1": (79,236,283,367),
    "sites_minehead_cam1": (277,259,591,502),
    "sites_mineheadgate_cam1": (234,243,527,476),
    "sites_motcombe_cam1": (350,424,635,597),
    "sites_northdrain_cam1": (306,319,406,465),
    "sites_pilloutfall_cam1": (83,118,272,329),
    "sites_sheptonmallet_cam1": (481,153,786,292),
    "sites_sheptonmallet_cam2": (504,556,1258,792),
    "sites_sistontunnel_cam1": (280,267,474,341),
    "sites_southperrott_cam1": (280,359,367,504),
    "sites_stannesscreen_cam1": (171,205,311,479),
    "sites_uphill_cam1": (332,294,645,435),
}

extracted_folder = "blockagedetection_dataset"
dataset_location = sys.argv[1]
output_location = sys.argv[2]
cropped_images_zip = "all_cropped_images.zip"

def crop(image_path, camera, image_name, classification):
    img = Image.open(image_path)
    crop_img = img.crop(coord_dict[camera])
    if "cropped_images" not in os.listdir(output_location):
        os.mkdir(f"{output_location}/cropped_images")
    if camera not in os.listdir(f"{output_location}/cropped_images/"):
        os.mkdir(f"{output_location}/cropped_images/{camera}")
    if classification not in os.listdir(f"{output_location}/cropped_images/{camera}"):
        os.mkdir(f"{output_location}/cropped_images/{camera}/{classification}")
    crop_img.save(f"{output_location}/cropped_images/{camera}/{classification}/{image_name}")

if extracted_folder not in os.listdir(dataset_location):
    with ZipFile(dataset_location, 'r') as zip_folder: ## only extract the dataset if not already extracted
        zip_folder.extractall(f"{dataset_location}/{extracted_folder}")
        
for camera in os.listdir(f"{dataset_location}/{extracted_folder}/images"):
    clear_images = os.listdir(f"{dataset_location}/{extracted_folder}/images/{camera}/clear")
    blocked_images = os.listdir(f"{dataset_location}/{extracted_folder}/images/{camera}/blocked")
    print(f"Starting: {camera}")
    for image in clear_images:
        crop(f"{dataset_location}/{extracted_folder}/images/{camera}/clear/{image}", camera, image, "clear")
    for image in blocked_images:
        crop(f"{dataset_location}/{extracted_folder}/images/{camera}/blocked/{image}", camera, image, "blocked")
    print(f"{camera} complete")

### re-zip the cropped_images directory once done to make file transfer across to Windows as fast as possible
# if "cropped_images" in os.listdir(output_location) and len(os.listdir(f"{output_location}/cropped_images")) > 0:
#     with ZipFile(f"{output_location}/{cropped_images_zip}", 'w') as output_zip:
#         for root, folders, files in os.walk(f"{output_location}/cropped_images"):
#             for folder_name in folders:
#                 absolute_path = os.path.join(root, folder_name)
#                 output_zip.write(absolute_path)
#             for file_name in files:
#                 absolute_path = os.path.join(root, file_name)
#                 output_zip.write(absolute_path)