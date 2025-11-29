import requests
from bs4 import BeautifulSoup

# Google Search Tool
class GoogleSearchTool:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search(self, query: str, num_results: int = 5) -> str:
        """
        Perform Google search and return search results (not URLs)
        """
        try:
            # Construct the Google search URL
            search_url = f"https://www.google.com/search?q={query}&num={num_results}"
            
            # Make the request
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            for i, result in enumerate(soup.find_all('div', class_='g')[:num_results]):
                # Extract title
                title_elem = result.find('h3')
                title = title_elem.get_text() if title_elem else "No title"
                
                # Extract snippet
                snippet_elem = result.find('span', class_='aCOpRe')
                if not snippet_elem:
                    snippet_elem = result.find('span', class_='st')
                snippet = snippet_elem.get_text() if snippet_elem else "No snippet"
                
                # Extract URL
                url_elem = result.find('a')
                url = url_elem['href'] if url_elem and 'href' in url_elem.attrs else "No URL"
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': url
                })
            
            # Format results as text
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}\n"
                )
            
            return "\n".join(formatted_results) if formatted_results else "No results found"
            
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def __str__(self):
        return "GoogleSearchTool"


# Create Google search tool instance
google_search_tool = GoogleSearchTool()