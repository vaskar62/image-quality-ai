from cleanvision import Imagelab

# Path to folder with images
image_folder = "../image_files"

# Initialize and run CleanVision
imagelab = Imagelab(data_path=image_folder)
imagelab.find_issues()
imagelab.report()

print("Report generated as 'cleanvision_report.html'")
