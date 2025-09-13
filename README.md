# Employee Recommendation System

This project implements an **employee recommendation system** using machine learning techniques to analyze employee performance and skills. The system provides recommendations for employees based on their skills and performance metrics, facilitating better task assignments and team management.

---

## Project Structure

```
sprint_planning
├── app
│   ├── models
│   │   └── Model_training.ipynb    
│   ├── routes
│   │   └── dashboard.html          
│   ├── static                      
│   ├── templates                  
│   └── __init__.py                
├── requirements.txt                
└── README.md                       
```

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/ritukanchi/sprint_planning>
cd sprint_planning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Follow these steps to set up and use the employee recommendation system:

1. **Set up the database**  
   ```bash
   python app/models/database.py
   ```

2. **Train the machine learning model**  
   ```bash
   python app/models/Model_training.py
   ```

3. **Start the API/server**  
   ```bash
   python api.py
   ```

4. **Access the dashboard**  
   Open the `dashboard.html` file in your local browser to interact with the system's user interface.

---

## Features

- **Employee Recommendations**: Recommends employees based on skills and performance metrics.  
- **User Interface**: Clean and interactive dashboard for task input and viewing recommendations.  
- **Notifications**: Real-time updates on sprint activities and email communications.  

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
