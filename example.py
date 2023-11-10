# To run this example please download the PascalVOC 2007 dataset first:
#
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# tar -xvf VOCtrainval_06-Nov-2007.tar

from pathlib import Path

from labelformat.formats import PascalVOCObjectDetectionInput

from lightly_insights import analyze, present

# Analyze an image folder.
image_analysis = analyze.analyze_images(
    image_folder=Path("./VOCdevkit/VOC2007/JPEGImages")
)

# Analyze object detections.
label_input = PascalVOCObjectDetectionInput(
    input_folder=Path("./VOCdevkit/VOC2007/Annotations"),
    category_names=(
        "person,bird,cat,cow,dog,horse,sheep,aeroplane,bicycle,boat,bus,car,"
        "motorbike,train,bottle,chair,diningtable,pottedplant,sofa,tvmonitor"
    ),
)
od_analysis = analyze.analyze_object_detections(label_input=label_input)

# Create HTML report.
present.create_html_report(
    output_folder=Path("./html_report"),
    image_analysis=image_analysis,
    od_analysis=od_analysis,
)
