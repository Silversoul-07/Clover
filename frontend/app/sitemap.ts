import fs from 'fs';
import path from 'path';

const baseUrl = 'http://localhost:3000';
const baseDir = 'app';
const excludeDirs = ['api'];

function getRoutes(dirPath = baseDir, routePath = ''): string[] {
  const fullPath = path.join(process.cwd(), dirPath);
  const entries = fs.readdirSync(fullPath, { withFileTypes: true });
  let routes: string[] = [];

  entries.forEach(entry => {
    if (entry.isDirectory()) {
      if (excludeDirs.includes(entry.name)) {
        return;
      }

      let newRoutePath = routePath;

      if (entry.name.startsWith('(')) {
        // It's a route group; traverse without adding to routePath
        routes = routes.concat(getRoutes(path.join(dirPath, entry.name), newRoutePath));
      } else {
        // Regular directory; append to routePath
        newRoutePath = `${routePath}/${entry.name}`;
        routes = routes.concat(getRoutes(path.join(dirPath, entry.name), newRoutePath));
      }
    } else if (entry.isFile() && entry.name === 'page.tsx') {
      // Add the current routePath as a valid route
      routes.push(routePath || '/');
    }
  });

  return routes;
}

function sitemap() {
  const routes = getRoutes();

  return routes.map(route => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date(),
    changeFrequency: 'weekly',
    priority: 1.0,
  }));
}

export default sitemap;