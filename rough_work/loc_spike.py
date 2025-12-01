"""
Spike script to test LoC document access patterns.
Run: python loc_spike.py
"""

import httpx
import asyncio
import json
from dataclasses import dataclass

@dataclass
class LocDocument:
    name: str
    url: str
    url_type: str  # item, resource, exhibits

LOC_DOCUMENTS = [
    LocDocument("Election Night 1860", "https://www.loc.gov/item/mal0440500/", "item"),
    LocDocument("Fort Sumter Decision", "https://www.loc.gov/resource/mal.0882800/", "resource"),
    LocDocument("Gettysburg Address", "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html", "exhibits"),
    LocDocument("Second Inaugural", "https://www.loc.gov/item/mal4361300/", "item"),
    LocDocument("Last Public Address", "https://www.loc.gov/resource/mal.4361800/", "resource"),
]

async def test_json_api(client: httpx.AsyncClient, doc: LocDocument) -> dict:
    """Test JSON API access for item/resource URLs."""
    result = {"name": doc.name, "url": doc.url, "type": doc.url_type}
    
    if doc.url_type == "exhibits":
        result["api_available"] = False
        result["note"] = "Exhibits require HTML scraping"
        return result
    
    json_url = doc.url.rstrip("/") + "/?fo=json"
    try:
        resp = await client.get(json_url, follow_redirects=True)
        result["status"] = resp.status_code
        
        if resp.status_code == 200:
            data = resp.json()
            result["api_available"] = True
            
            # Check for transcript
            item = data.get("item", {})
            resources = data.get("resources", [])
            
            # Look for transcript info
            result["has_transcript"] = False
            result["transcript_url"] = None
            
            # Check resources for text/transcript
            for resource in resources:
                if "fulltext_file" in resource:
                    result["has_transcript"] = True
                    result["transcript_url"] = resource["fulltext_file"]
                if "pdf" in resource and resource["pdf"]:
                    result["pdf_url"] = resource["pdf"]
                    
            # Check for online text format
            online_format = item.get("online_format", [])
            if "online text" in online_format:
                result["has_online_text"] = True
                
            # Get title
            result["title"] = item.get("title", "Unknown")
            
            # Look for transcript in different locations
            if "transcript" in str(data).lower():
                result["transcript_mentioned"] = True
                
        else:
            result["api_available"] = False
            result["error"] = f"HTTP {resp.status_code}"
            
    except Exception as e:
        result["api_available"] = False
        result["error"] = str(e)
    
    return result

async def test_exhibits_html(client: httpx.AsyncClient, doc: LocDocument) -> dict:
    """Test HTML access for exhibits URLs."""
    result = {"name": doc.name, "url": doc.url, "type": doc.url_type}
    
    try:
        resp = await client.get(doc.url, follow_redirects=True)
        result["status"] = resp.status_code
        
        if resp.status_code == 200:
            result["html_available"] = True
            result["content_length"] = len(resp.text)
            
            # Check if it contains transcript text
            if "four score" in resp.text.lower():
                result["has_transcript_content"] = True
            else:
                result["has_transcript_content"] = False
                
    except Exception as e:
        result["html_available"] = False
        result["error"] = str(e)
    
    return result

async def main():
    print("=" * 60)
    print("LoC Document Access Spike")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = []
        
        for doc in LOC_DOCUMENTS:
            print(f"\nTesting: {doc.name}")
            print(f"  URL: {doc.url}")
            
            # Rate limit: wait 3 seconds between requests (20/min limit)
            await asyncio.sleep(3)
            
            if doc.url_type == "exhibits":
                result = await test_exhibits_html(client, doc)
            else:
                result = await test_json_api(client, doc)
            
            results.append(result)
            
            # Print summary
            if result.get("api_available") or result.get("html_available"):
                print(f"  ✓ Accessible")
                if result.get("has_transcript"):
                    print(f"  ✓ Transcript found: {result.get('transcript_url')}")
                elif result.get("has_transcript_content"):
                    print(f"  ✓ Transcript in HTML")
                elif result.get("has_online_text"):
                    print(f"  ~ Has online text format")
                else:
                    print(f"  ✗ No transcript found in response")
            else:
                print(f"  ✗ Not accessible: {result.get('error', 'Unknown')}")
        
        # Save full results
        print("\n" + "=" * 60)
        print("Full Results (JSON)")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
        # Save to file
        with open("loc_spike_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to loc_spike_results.json")

if __name__ == "__main__":
    asyncio.run(main())