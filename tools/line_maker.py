import matplotlib.pyplot as plt

def generate_line_chart(chart_data, save_path):
    try:
        # Extract data for the primary chart
        data = chart_data.get("data", [])
        if not data:
            raise ValueError("The 'data' field is missing or empty.")

        labels = [entry["label"] for entry in data]
        values = [entry["value"] for entry in data]
        
        # Extract metadata for the chart
        title = chart_data.get("title", "Line Chart")
        x_axis_label = chart_data.get("xAxisLabel", "X-Axis")
        y_axis_label = chart_data.get("yAxisLabel", "Y-Axis")
        
        # Create the line chart
        plt.figure(figsize=(8, 5))
        plt.plot(labels, values, marker="o", color="blue")
        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.grid()
        
        # Save the chart to the specified path
        plt.savefig(save_path)
        plt.close()
        return save_path
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None


chart_data = {
  "chartType": "line",
  "title": "Apple's Operating Expenses and Revenue Over the Past Five Years",
  "xAxisLabel": "Fiscal Year",
  "yAxisLabel": "Amount in Billion USD",
  "data": [
    {
      "label": "2020",
      "value": 208.2
    },
    {
      "label": "2021",
      "value": 274.891
    },
    {
      "label": "2022",
      "value": 274.9
    },
    {
      "label": "2023",
      "value": 268.984
    },
    {
      "label": "2024",
      "value": 267.819
    }
  ],
  "operatingExpensesPercentage": [
    {
      "label": "2020",
      "value": (208.2 / 274.9) * 100
    },
    {
      "label": "2021",
      "value": (274.891 / 274.9) * 100
    },
    {
      "label": "2022",
      "value": (274.9 / 274.9) * 100
    },
    {
      "label": "2023",
      "value": (268.984 / 274.9) * 100
    },
    {
      "label": "2024",
      "value": (267.819 / 94.93) * 100
    }
  ]
}

# Path to save the chart
save_path = "line_chart.png"
generate_line_chart(chart_data, save_path)