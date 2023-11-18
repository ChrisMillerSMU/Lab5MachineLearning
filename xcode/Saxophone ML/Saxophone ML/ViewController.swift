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
    
    // Model to retain stopwatch time for train button
    lazy var trainTimer: TimerModel = {
        return TimerModel()
    }()
    
    // Model to retain stopwatch time for test button
    lazy var testTimer: TimerModel = {
        return TimerModel()
    }()
    
    var timer: Timer?
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        setupPopUpButton()
    }
    
    @IBAction func switchNames(_ sender: Any?) {
        print("switched")
    }
    
    @IBAction func trainButtonPressed(_ sender: UIButton) {
        self.testButton.isEnabled = false
        self.trainButton.isEnabled = false
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
        self.trainTimer.setRemainingTime(withInterval: 300)
        self.timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
            self.updateTimer(button:self.trainButton, stopwatch:self.trainTimer)
        }
        
    }
    
    @IBAction func testButtonPressed(_ sender: UIButton) {
        //Straight to backend
        self.testButton.isEnabled = false
        self.trainButton.isEnabled = false
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
        //Use currentName and currentModel as params
        self.testTimer.setRemainingTime(withInterval: 300)
        self.timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
            self.updateTimer(button:self.testButton, stopwatch:self.testTimer)
        }
    }
    
    func updateTimer(button:UIButton, stopwatch:TimerModel){
        if stopwatch.getRemainingTime() > 0 {
            stopwatch.decrementRemainingTime()
            stopwatch.changeDisplay()
            button.titleLabel?.text = stopwatch.timeDisplay
        }
        else{
            // Stop timer with finished set to TRUE, since it got to 0
            self.stopTimer(button:button)
        }
    }
    
    func stopTimer(button:UIButton) {
        self.timer?.invalidate()
        self.timer = nil
        self.trainButton.titleLabel?.text = "Train"
        self.testButton.titleLabel?.text = "TEST"
        button.backgroundColor = .red
        recordInput()
        self.timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            self.stopRecord(button:button)
        }
    }
    
    func recordInput() {
        //code to record audio data
        
    }
    
    func stopRecord(button:UIButton) {
        //code to send off data
        
        
        
        self.timer?.invalidate()
        self.timer = nil
        self.testButton.isEnabled = true
        self.trainButton.isEnabled = true
        button.backgroundColor = nil
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

