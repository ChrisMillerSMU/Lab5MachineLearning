//
//  ViewController.swift
//  Saxophone ML
//
//  Created by Reece Iriye on 11/15/23.
//

import UIKit

class ViewController: UIViewController {
    
    
    @IBOutlet weak var modelSegmentedSwitch: UISegmentedControl!
    
    @IBOutlet weak var trainingLabel: UILabel!
    
    @IBOutlet weak var nameDropdownButton: UIButton!
    
    @IBOutlet weak var trainButton: UIButton!
    
    @IBOutlet weak var testButton: UIButton!
    
    @IBOutlet weak var predictionLabel: UILabel!
    
    @IBOutlet weak var evaluateModelsButton: UIButton!
    
    var modelType = "Spectogram CNN"
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        setupPopUpButton()
    }
    
    @IBAction func switchModels(_ sender: UISegmentedControl) {
        switch modelSegmentedSwitch.selectedSegmentIndex
        {
        case 0:
            print("switch to model1")
        case 1:
            print("switch to model2")
        default:
            break
        }
    }
    
    @IBAction func switchNames(_ sender: Any?) {
        print("switched")
    }
    
    @IBAction func trainButtonPressed(_ sender: UIButton) {
            //Straight to backend
        if let currentName = nameDropdownButton.titleLabel?.text {
            print(currentName)
        } else {
            print("Button text is nil")
        }
        
        if let currentModel = modelSegmentedSwitch.titleForSegment(at: modelSegmentedSwitch.selectedSegmentIndex) {
            print(currentModel)
        } else {
            print("Button text is nil")
        }
        
        //Use currentName and currentModel as params
        
    }
    
    @IBAction func testButtonPressed(_ sender: UIButton) {
        //Straight to backend
        if let currentName = nameDropdownButton.titleLabel?.text {
            print(currentName)
        } else {
            print("Button text is nil")
        }
    
        if let currentModel = modelSegmentedSwitch.titleForSegment(at: modelSegmentedSwitch.selectedSegmentIndex) {
            print(currentModel)
        } else {
            print("Button text is nil")
        }
    
        //Use currentName and currentModel as params
    }
    
    func setupPopUpButton() {
        let popUpButtonClosure = { (action: UIAction) in
            print("Pop-up action")
        }
                
        nameDropdownButton.menu = UIMenu(children: [
            UIAction(title: "Reece", handler: popUpButtonClosure),
            UIAction(title: "Ethan", handler: popUpButtonClosure),
            UIAction(title: "Rafe", handler: popUpButtonClosure),
            UIAction(title: "Chris", handler: popUpButtonClosure)
        ])
        nameDropdownButton.showsMenuAsPrimaryAction = true
    }
    
}

