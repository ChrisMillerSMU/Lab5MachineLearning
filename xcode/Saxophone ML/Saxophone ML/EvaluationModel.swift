//
//  EvaluationModel.swift
//  Saxophone ML
//
//  Created by Ethan Haugen on 11/19/23.
//

import UIKit

class EvaluationModel: NSObject {
    private var modelString = "Spectrogram CNN"
    private var spectogramAccuracyString = "--.-"
    private var logisticAccuracyString = "--.-"
    private var labelString = "Training accuracy for Spectogram CNN is: --.-%"
    
    func setModel(model:String) {
        modelString = model
    }
    
    func setAccuracy(accuracy:String, myCase:Int) {
        switch(myCase){
        case 0:
            self.spectogramAccuracyString = accuracy + "%"
            break
        case 1:
            self.logisticAccuracyString = accuracy + "%"
            break
        default:
            break
        }
    }
    
    func updateLabelString() {
        switch(self.modelString){
        case "Spectrogram CNN":
            labelString = "Training accuracy for \(self.modelString) is: \(self.spectogramAccuracyString)"
        case "Logistic Regression":
            labelString = "Training accuracy for \(self.modelString) is: \(self.logisticAccuracyString)"
        default:
            break
        }
        
        return labelString
        
    }
}
