//
//  ViewController.swift
//  Saxophone ML
//
//  Created by Reece Iriye on 11/15/23.
//

let SERVER_URL = "http://10.8.159.212:8000"

import UIKit

class ViewController: UIViewController, URLSessionDelegate {
    
    struct PredictionRequest: Codable {
        let raw_audio: [Float]
        let ml_model_type: String
    }
    
    struct TrainingRequest: Codable {
        let raw_audio: [Float]
        let audio_label: String
        let ml_model_type: String
    }
    
    // MARK: Class Properties
    lazy var session: URLSession = {
        let sessionConfig = URLSessionConfiguration.ephemeral
        
        sessionConfig.timeoutIntervalForRequest = 5.0
        sessionConfig.timeoutIntervalForResource = 8.0
        sessionConfig.httpMaximumConnectionsPerHost = 1
        
        return URLSession(configuration: sessionConfig,
            delegate: self,
            delegateQueue:self.operationQueue)
    }()
    
    let operationQueue = OperationQueue()
    
    @IBOutlet weak var modelSegmentedSwitch: UISegmentedControl!
    
    @IBOutlet weak var trainingLabel: UILabel!
    
    @IBOutlet weak var nameDropdownButton: UIButton!
    
    @IBOutlet weak var trainButton: UIButton!
    
    @IBOutlet weak var testButton: UIButton!
    
    @IBOutlet weak var predictionLabel: UILabel!
    
    @IBOutlet weak var accuracyLabel: UILabel!
    
    // Model to retain stopwatch time for train button
    lazy var trainTimer: TimerModel = {
        return TimerModel()
    }()
    
    // Model to retain stopwatch time for test button
    lazy var testTimer: TimerModel = {
        return TimerModel()
    }()
    
    // Model to retain stopwatch time for test button
    lazy var evaluationScores: EvaluationModel = {
        return EvaluationModel()
    }()
    
    var timer: Timer?
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        setupPopUpButton()
    }
    
    @IBAction func switchNames(_ sender: UISegmentedControl) {
        if let currentModel = modelSegmentedSwitch.titleForSegment(at: modelSegmentedSwitch.selectedSegmentIndex) {
            evaluationScores.setModel(model: currentModel)
        } else {
            print("Button text is nil")
        }
    }
    
    @IBAction func trainButtonPressed(_ sender: UIButton) {
        self.testButton.isEnabled = false
        self.trainButton.isEnabled = false
        self.modelSegmentedSwitch.isEnabled = false
        self.nameDropdownButton.isEnabled = false
        
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
        self.modelSegmentedSwitch.isEnabled = false
        self.nameDropdownButton.isEnabled = false
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
        // start storing to an array in our audio model
    }
    
    func stopRecord(button:UIButton) {
        //code to send off data
        
        var postType = button.titleLabel?.text
        var currentName = ""
        //Straight to backend
        if let name = nameDropdownButton.titleLabel?.text {
            currentName = name
        } else {
            print("Button text is nil")
        }
        
        
        var currentModel = evaluationScores.getModel()
        
        // THIS IS WHERE WE SEND UP THE DATA
        
        if(postType == "TEST"){
            sendPredictionPostRequest(model:currentModel)
        }
        else{
            sendTrainingPostRequest(model:currentModel, label: currentName)
        }
        
        self.timer?.invalidate()
        self.timer = nil
        self.testButton.isEnabled = true
        self.trainButton.isEnabled = true
        self.modelSegmentedSwitch.isEnabled = true
        self.nameDropdownButton.isEnabled = true
        button.backgroundColor = nil
    }
    
    func sendPredictionPostRequest(model: String) {
        let data = [Float](repeating: 0.001, count: 100) // Make sure this is [Float], not [Double]
        
        let baseURL = "\(SERVER_URL)/predict_one"
        guard let postUrl = URL(string: baseURL) else { return }
        
        var request = URLRequest(url: postUrl)
        
        let predictionRequest = PredictionRequest(raw_audio: data, ml_model_type: model)
        
        do {
            let requestBody = try JSONEncoder().encode(predictionRequest)
            request.httpMethod = "POST"
            request.httpBody = requestBody
            request.addValue("application/json", forHTTPHeaderField: "Content-Type") // Set content type
            
            let postTask = URLSession.shared.dataTask(with: request) { (data, response, error) in
                if let error = error {
                    print("Error:", error)
                    return
                }
                
                guard let data = data else {
                    print("No data")
                    return
                }
                
                do {
                    if let jsonDictionary = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                        print(jsonDictionary)
                        if let labelResponse = jsonDictionary["audio_prediction"] as? String {
                            DispatchQueue.main.async {
                                self.predictionLabel.text = labelResponse
                            }
                        }
                    }
                } catch {
                    print("Error decoding JSON:", error)
                }
            }
            
            postTask.resume() // Start the task
            
        } catch {
            print("Error encoding JSON:", error)
        }
    }
    
    func sendTrainingPostRequest(model: String, label: String) {
        let data = [Float](repeating: 0.001, count: 100) // Use [Float] to match the expected Pydantic model

        let baseURL = "\(SERVER_URL)/upload_labeled_datapoint_and_update_model"
        guard let postUrl = URL(string: baseURL) else { return }

        var request = URLRequest(url: postUrl)

        let trainingRequest = TrainingRequest(raw_audio: data, audio_label: label, ml_model_type: model)

        do {
            let requestBody = try JSONEncoder().encode(trainingRequest)
            request.httpMethod = "POST"
            request.httpBody = requestBody
            request.addValue("application/json", forHTTPHeaderField: "Content-Type") // Set content type

            let postTask = URLSession.shared.dataTask(with: request) { (data, response, error) in
                if let error = error {
                    print("Error:", error)
                    return
                }

                guard let data = data else {
                    print("No data")
                    return
                }

                do {
                    if let jsonDictionary = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                        print(jsonDictionary)
                        if let labelResponse = jsonDictionary["resub_accuracy"] as? String {
                            DispatchQueue.main.async {
                                self.accuracyLabel.text = labelResponse
                            }
                        }
                    }
                } catch {
                    print("Error decoding JSON:", error)
                }
            }

            postTask.resume() // Start the task

        } catch {
            print("Error encoding JSON:", error)
        }
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
    
    //MARK: JSON Conversion Functions
    func convertDictionaryToData(with jsonUpload:NSDictionary) -> Data?{
        do { // try to make JSON and deal with errors using do/catch block
            let requestBody = try JSONSerialization.data(withJSONObject: jsonUpload, options:JSONSerialization.WritingOptions.prettyPrinted)
            return requestBody
        } catch {
            print("json error: \(error.localizedDescription)")
            return nil
        }
    }
    
    func convertDataToDictionary(with data:Data?)->NSDictionary{
        do { // try to parse JSON and deal with errors using do/catch block
            let jsonDictionary: NSDictionary =
                try JSONSerialization.jsonObject(with: data!,
                                              options: JSONSerialization.ReadingOptions.mutableContainers) as! NSDictionary
            
            return jsonDictionary
            
        } catch {
            
            if let strData = String(data:data!, encoding:String.Encoding(rawValue: String.Encoding.utf8.rawValue)){
                            print("printing JSON received as string: "+strData)
            }else{
                print("json error: \(error.localizedDescription)")
            }
            return NSDictionary() // just return empty
        }
    }
    
}

