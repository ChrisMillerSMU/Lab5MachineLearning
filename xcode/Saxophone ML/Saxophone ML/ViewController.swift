//
//  ViewController.swift
//  Saxophone ML
//
//  Created by Reece Iriye on 11/15/23.
//

import UIKit

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    @IBAction func switchModels(_ sender: UISegmentedControl) {
    }
    
    @IBOutlet weak var modelSegmentedSwitch: UISegmentedControl!
    
    @IBOutlet weak var trainingLabel: UILabel!
    
    @IBOutlet weak var nameDropdownButton: UIButton!
    
    @IBAction func switchNames(_ sender: UIButton) {
    }
    
    @IBAction func trainButtonPressed(_ sender: UIButton) {
    }
    
    @IBAction func testButtonPressed(_ sender: UIButton) {
    }
    @IBOutlet weak var predictionLabel: UILabel!
    
    
    
}

