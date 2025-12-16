import { createBrowserRouter } from 'react-router-dom';
import { Layout } from '@/layout';
import Home from '@/pages/Home';
import Dashboard from '@/pages/Dashboard';
import Scanner from '@/pages/Scanner';
import Models from '@/pages/Models';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: 'dashboard',
        element: <Dashboard />,
      },
      {
        path: 'scanner',
        element: <Scanner />,
      },
      {
        path: 'models',
        element: <Models />,
      },
    ],
  },
]);
