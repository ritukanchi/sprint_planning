# Employee Recommendation System

This project implements an employee recommendation system using machine learning techniques to analyze employee performance and skills. The system provides recommendations for employees based on their skills and performance metrics, facilitating better task assignments and team management.

## Project Structure

```
sprint_planning
├── app
│   ├── models
│   │   └── Model_training.ipynb  # Jupyter notebook for model training and recommendations
│   ├── routes
│   │   └── dashboard.html         # HTML file for the user interface
│   ├── static                     # Directory for static files (JS/CSS)
│   ├── templates                  # Directory for dynamic HTML templates
│   └── __init__.py               # Application initialization file
├── requirements.txt               # List of project dependencies
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sprint_planning
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Launch the application:
   - If using Flask, run the following command:
     ```
     flask run
     ```

2. Open your web browser and navigate to `http://127.0.0.1:5000` to access the dashboard.

3. Use the dashboard to enter task skills and receive employee recommendations based on the trained models.

## Features

- **Employee Recommendations**: The system recommends employees based on their skills and performance metrics.
- **User Interface**: A clean and interactive dashboard for users to input tasks and view recommendations.
- **Notifications**: Real-time updates on sprint activities and email communications.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.