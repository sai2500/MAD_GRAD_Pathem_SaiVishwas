//
//  ViewController.swift
//  Mad_Grad_Pathem_SaiVishwas
//
//  Created by Sai Vishwas Pathem on 4/27/24.
//

import UIKit
import Vision
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classificationLabel: UILabel!
    
    // Action for selecting an image
    @IBAction func selectImageTapped(_ sender: UIButton) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.mediaTypes = ["public.image"]

        let actionSheet = UIAlertController(title: "Select Image", message: nil, preferredStyle: .actionSheet)
        
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            actionSheet.addAction(UIAlertAction(title: "Camera", style: .default, handler: { [weak self] _ in
                imagePickerController.sourceType = .camera
                self?.present(imagePickerController, animated: true, completion: nil)
            }))
        }
        
        actionSheet.addAction(UIAlertAction(title: "Photo Library", style: .default, handler: { [weak self] _ in
            imagePickerController.sourceType = .photoLibrary
            self?.present(imagePickerController, animated: true, completion: nil)
        }))
        
        actionSheet.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(actionSheet, animated: true, completion: nil)
    }


    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: MyImageClassifier2(configuration: MLModelConfiguration()).model)
            return VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()


    override func viewDidLoad() {
        super.viewDidLoad()
    }

    @IBAction func selectImage(_ sender: UIButton) {
        let pickerController = UIImagePickerController()
        pickerController.delegate = self
        pickerController.sourceType = .photoLibrary
        present(pickerController, animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        guard let image = info[.originalImage] as? UIImage else { return }
        imageView.image = image
        updateClassifications(for: image)
    }

    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."
        let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))!
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        #if targetEnvironment(simulator)
        if #available(iOS 17.0, *) {
            let allDevices = MLComputeDevice.allComputeDevices

            for device in allDevices {
                if device.description.contains("MLCPUComputeDevice") {
                    classificationRequest.setComputeDevice(.some(device), for: .main)
                    break
                }
            }

        } else {
            classificationRequest.usesCPUOnly = true
        }
        #endif
        
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Error: \(error)") // Changed print statement
            }
        }
    }

    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let strongSelf = self else { return }

            if let error = error {
                strongSelf.classificationLabel.text = "Classification failed: \(error.localizedDescription)"
                return
            }

            guard let results = request.results else {
                strongSelf.classificationLabel.text = "Unable to classify image."
                return
            }

            let classifications = results as! [VNClassificationObservation]
            if classifications.isEmpty {
                strongSelf.classificationLabel.text = "Nothing recognized."
            } else {
                let topClassification = classifications.max(by: { a, b in a.confidence < b.confidence })
                if let classification = topClassification {
                    strongSelf.classificationLabel.text = "Classification: \(classification.identifier)"
                } else {
                    strongSelf.classificationLabel.text = "Classification: Unknown"
                }
            }
        }
    }
}
