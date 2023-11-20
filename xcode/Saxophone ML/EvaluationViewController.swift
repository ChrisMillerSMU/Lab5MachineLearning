//
//  EvaluationViewController.swift
//  Saxophone ML
//
//  Created by Ethan Haugen on 11/18/23.
//

import UIKit

class EvaluationViewController: UIViewController {
    
    @IBOutlet weak var nameSwitch: UIButton!
    @IBOutlet weak var spectogramScore: UILabel!
    @IBOutlet weak var XGBoostScore: UILabel!
    @IBOutlet weak var recordDataButton: UIButton!
    
    lazy var testTimer: TimerModel = {
        return TimerModel()
    }()
    
    var timer: Timer?
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        setupPopUpButton()
    }
    
    @IBAction func recordDataButton(_ sender: UIButton) {
            // code to record then send to server to be saved
        self.recordDataButton.isEnabled = false
        if let currentName = nameSwitch.titleLabel?.text {
            print(currentName)
        } else {
            print("Button text is nil")
        }
    
        //Use currentName and currentModel as params
        self.testTimer.setRemainingTime(withInterval: 300)
        self.timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
            self.updateTimer(button:self.recordDataButton, stopwatch:self.testTimer)
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
        self.recordDataButton.titleLabel?.text = "Record Testing Data"
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
        self.recordDataButton.isEnabled = true
        button.backgroundColor = nil
    }
    
    @IBAction func refreshScores(_ sender: Any) {
            // code to refresh scores from server
    }
    
    
    func setupPopUpButton() {
        let popUpButtonClosure = { (action: UIAction) in
            print("Pop-up action")
        }
                
        nameSwitch.menu = UIMenu(children: [
            UIAction(title: "Reece", handler: popUpButtonClosure),
            UIAction(title: "Ethan", handler: popUpButtonClosure),
            UIAction(title: "Rafe", handler: popUpButtonClosure),
            UIAction(title: "Chris", handler: popUpButtonClosure)
        ])
        nameSwitch.showsMenuAsPrimaryAction = true
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
