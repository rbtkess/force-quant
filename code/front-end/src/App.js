import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import {
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Box,
  Tabs,
  Tab,
  List,
  ListItem,
} from '@mui/material';
import {
  Chart as ChartJS,
  RadialLinearScale,
  ArcElement,
  LineElement,
  PointElement,
  Filler,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale,
} from 'chart.js';
import { Radar, Line } from 'react-chartjs-2';

ChartJS.register(
  RadialLinearScale,
  ArcElement,
  LineElement,
  PointElement,
  Filler,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale
);

const socket = io('http://localhost:5000');

function App() {
  const [inputText, setInputText] = useState('');
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progressMessages, setProgressMessages] = useState([]);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    // Listen for progress updates from the server
    socket.on('progress', (message) => {
      setProgressMessages((prevMessages) => [...prevMessages, message.data]);
    });

    // Cleanup function to remove the listener when the component unmounts
    return () => {
      socket.off('progress');
    };
  }, []);

  const handleChange = (e) => {
    setInputText(e.target.value);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleClick = async () => {
    setLoading(true);
    setError(null);
    setOutput(null);
    setProgressMessages([]);
    try {
      const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: inputText }),
      });

      const data = await response.json();
      setProgressMessages([]);
      setOutput(data);
      console.log(data);
    } catch (error) {
      setError(error.message);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Parse the summary JSON string if output is available
  const summary = output ? JSON.parse(output.summary).forces : [];

  // Function to convert newlines to <br/> tags
  const convertNewlinesToBreaks = (text) => {
    return text.replace(/\n/g, '<br/>');
  };

  // Define radar chart data
  const radarChartData = summary.length > 0
    ? {
        labels: summary.map((force) => force.force),
        datasets: [
          {
            label: output.ticker,
            data: summary.map((force) => force.score),
            fill: true,
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgb(255, 99, 132)',
            pointBackgroundColor: 'rgb(255, 99, 132)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(255, 99, 132)',
          },
        ],
      }
    : {};

  // Define radar chart options
  const radarChartOptions = {
    maintainAspectRatio: false,
    scales: {
      r: {
        min: 0,
        max: 10,
        ticks: {
          stepSize: 2,
        },
        angleLines: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        pointLabels: {
          font: {
            size: 11,
          },
        },
      },
    },
    plugins: {
      legend: {
        position: 'bottom',
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            return `${context.dataset.label}: ${context.raw}`;
          },
        },
      },
      title: {
        display: true,
        text: "Porter's 5 Forces", 
        font: {
          size: 16,
        },
      },
    },
  };

  // // Line Chart data
  // const lineChartData = {
  //   labels: Array.from({ length: 6 * 365 }, (_, i) => `Day ${i + 1}`), // Placeholder days
  //   datasets: [
  //     {
  //       label: 'History',
  //       data: Array.from({ length: 5 * 365 }, () => Math.random() * 100 + 150), // Random data
  //       fill: false,
  //       borderColor: 'rgb(75, 192, 192)',
  //       tension: 0.1,
  //     },
  //     {
  //       label: 'Forecast',
  //       data: Array.from({ length: 365 }, () => Math.random() * 50 + 100), // Random forecast data
  //       fill: false,
  //       borderColor: 'rgb(255, 205, 86)',
  //       tension: 0.1,
  //     },
  //   ],
  // };
  const lineChartData = output ? {
    labels: output.stock_prices.labels,
    datasets: output.stock_prices.datasets.map((dataset, index) => ({
      ...dataset,
      pointRadius: 0, // Remove the dots
      backgroundColor: index === 0 ? 'rgba(34, 139, 34, 0.2)' : 'rgba(0, 0, 255, 0.2)',
      borderColor: index === 0 ? 'rgba(34, 139, 34, 1)' : 'rgba(0, 0, 255, 1)', // Green for history, blue for forecast
    })),
  } : {
    labels: [],
    datasets: [],
  };

  // Define line chart options
  const lineChartOptions = {
    maintainAspectRatio: false, // Allows dynamic sizing
    plugins: {
      legend: {
        position: 'bottom', // Place legend at the bottom
      },
      title: {
        display: true,
        text: 'Stock Price', // Title for the line chart
        font: {
          size: 16,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Price',
        },
      },
    },
  };

  return (
    <Container maxWidth="lg" style={{ marginTop: '2rem' }}>
      <Paper elevation={3} style={{ padding: '2rem' }}>
        <Typography variant="h4" align="center" color="primary" gutterBottom>
          Force-Quant
        </Typography>

        {/* Disclaimer Text */}
        <Typography
          variant="body2"
          align="center"
          style={{ color: 'red', fontSize: '0.875rem' }}
        >
          Academic Exercise. Not investment advice.
        </Typography>

        <Box display="flex" alignItems="center" mb={3}>
          <TextField
            label="ticker"
            variant="outlined"
            fullWidth
            value={inputText}
            onChange={handleChange}
            disabled={loading}
            margin="normal"
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleClick}
            disabled={loading}
            style={{ marginLeft: '1rem', height: '56px' }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              'Submit'
            )}
          </Button>
        </Box>

        {error && (
          <Typography color="error" align="center">
            Error: {error}
          </Typography>
        )}

        {progressMessages.length > 0 && (
          <Box mt={3}>
            <Typography variant="h6" gutterBottom>
              Progress:
            </Typography>
            <List>
              {progressMessages.map((msg, index) => (
                <ListItem key={index}>
                  <Typography
                    variant="body2"
                    style={{ fontSize: '0.875rem', lineHeight: '1.2' }} // Adjusted line height
                  >
                    {msg}
                  </Typography>
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {output && (
          <>
            {/* Company Title */}
            <Typography
              variant="h5"
              align="center"
              style={{ marginBottom: '1rem', color: 'grey' }}
            >
              {output.company_name}
            </Typography>

            {/* Charts */}
            <Box display="flex" justifyContent="space-between" mt={3}>
              {/* Radar Chart */}
              <Box width="55%" height="400px">
                <Radar data={radarChartData} options={radarChartOptions}/>
              </Box>

              {/* Line Chart */}
              <Box width="48%" height="400px">
                <Line data={lineChartData} options={lineChartOptions}/>
              </Box>
            </Box>

            {/* Tabbed Interface for Force Details */}
            <Box mt={3}>
              <Tabs value={tabValue} onChange={handleTabChange} centered>
                {summary.map((force, index) => (
                  <Tab key={index} label={force.force} />
                ))}
              </Tabs>

              {summary.map((force, index) => (
                <div
                  key={index}
                  role="tabpanel"
                  hidden={tabValue !== index}
                  style={{ padding: '1rem' }}
                >
                  {tabValue === index && (
                    <>
                      <Typography variant="h6">Summary</Typography>
                      <Typography
                        variant="body2"
                        style={{ marginBottom: '1rem' }}
                      >
                        {force.justification}
                      </Typography>
                      <Typography variant="h6">Detail</Typography>
                      <Typography
                        variant="body2"
                        dangerouslySetInnerHTML={{
                          __html: convertNewlinesToBreaks(
                            output.force_context[force.force]
                          ),
                        }}
                      />
                    </>
                  )}
                </div>
              ))}
            </Box>
          </>
        )}
      </Paper>
    </Container>
  );
}

export default App;
