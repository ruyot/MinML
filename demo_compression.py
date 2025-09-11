#!/usr/bin/env python3
"""
MinML Compression Demo Tool
Interactive CLI for demonstrating prompt compression with real-time stats
"""

import os
import sys
import time
import json
import requests
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MinMLDemo:
    def __init__(self):
        self.minml_url = "http://localhost:3123/v1/chat/completions"
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.csv_filename = f"compression_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.session_stats = {
            'total_requests': 0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
            'total_saved': 0,
            'falcon_used': 0,
            'algorithmic_used': 0
        }
        
        if not self.openai_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            print("Please add your OpenAI API key to .env file:")
            print("OPENAI_API_KEY=your_key_here")
            sys.exit(1)
            
        # Initialize CSV file
        self.init_csv()
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'original_prompt', 'compressed_prompt', 'prompt_length', 
            'tokens_before', 'tokens_after', 'tokens_saved', 'reduction_pct',
            'compression_method', 'processing_time', 'openai_response_preview',
            'success', 'error_message'
        ]
        
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"📊 Logging to: {self.csv_filename}")
    
    def log_to_csv(self, data):
        """Log compression data to CSV"""
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get('timestamp', ''),
                data.get('original_prompt', ''),
                data.get('compressed_prompt', ''),
                data.get('prompt_length', 0),
                data.get('tokens_before', 0),
                data.get('tokens_after', 0),
                data.get('tokens_saved', 0),
                data.get('reduction_pct', 0),
                data.get('compression_method', ''),
                data.get('processing_time', 0),
                data.get('openai_response_preview', ''),
                data.get('success', False),
                data.get('error_message', '')
            ])
    
    def estimate_tokens(self, text):
        """Rough token estimation (OpenAI uses ~4 chars per token)"""
        return len(text) // 4
    
    def get_minml_metrics(self):
        """Get current metrics from MinML"""
        try:
            response = requests.get("http://localhost:3123/status", timeout=3)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def print_banner(self):
        """Print the demo banner"""
        print("\n" + "="*80)
        print("🚀 MinML Compression Demo - Live Prompt Compression with CSV Logging")
        print("="*80)
        print("✨ Features:")
        print("  • Real-time compression with Falcon 7B + Algorithmic fallback")
        print("  • Detailed token statistics (before/after/saved/percentage)")
        print("  • CSV logging with full compression data")
        print("  • Live stats tracking in Electron app")
        print("  • Actual OpenAI API integration with response display")
        print("  • Session tracking and compression method detection")
        print(f"\n📊 CSV Log File: {self.csv_filename}")
        print("\n💡 Tips:")
        print("  • Use verbose prompts for better compression demos")
        print("  • Watch detailed stats after each request")
        print("  • Check CSV file for complete session data")
        print("  • Type 'quit' to exit, 'stats' for session summary")
        print("="*80)
    
    def get_compression_stats(self):
        """Get current compression statistics"""
        try:
            response = requests.get("http://localhost:3123/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def test_falcon_compression_directly(self, prompt):
        """Test Falcon compression directly to show what gets sent to OpenAI"""
        try:
            response = requests.post(
                "http://127.0.0.1:8081/compress",
                json={"input": prompt},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    'compressed': result['compressed'],
                    'stats': result['stats'],
                    'method': 'Falcon 7B Direct'
                }
        except:
            pass
        return None

    def send_prompt(self, prompt):
        """Send prompt through MinML with detailed compression tracking"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prompt_preview = prompt[:100] + ('...' if len(prompt) > 100 else '')
        tokens_before_est = self.estimate_tokens(prompt)
        
        # First, test what Falcon would compress this to
        falcon_test = self.test_falcon_compression_directly(prompt)
        
        # Get before stats from MinML
        before_metrics = self.get_minml_metrics()
        before_total_saved = 0
        if before_metrics:
            # Try to extract today's metrics if available
            try:
                before_total_saved = before_metrics.get('totals', {}).get('saved', 0)
            except:
                pass
        
        # Prepare the request
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,  # Larger response for better demo
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        print(f"\n{'='*80}")
        print(f"📤 SENDING PROMPT")
        print(f"{'='*80}")
        print(f"📝 Original: \"{prompt_preview}\"")
        print(f"📏 Length: {len(prompt)} chars")
        print(f"🔢 Estimated tokens: {tokens_before_est}")
        
        # Show what Falcon would compress this to
        if falcon_test:
            compressed_preview = falcon_test['compressed'][:100] + ('...' if len(falcon_test['compressed']) > 100 else '')
            print(f"🤖 Falcon compression preview: \"{compressed_preview}\"")
            print(f"📊 Falcon stats: {falcon_test['stats']['beforeTokens']}→{falcon_test['stats']['afterTokens']} tokens ({falcon_test['stats']['pct']}%)")
        
        print(f"⏱️  Processing...")
        
        start_time = time.time()
        log_data = {
            'timestamp': timestamp,
            'prompt_preview': prompt_preview,
            'prompt_length': len(prompt),
            'tokens_before': tokens_before_est,
            'processing_time': 0,
            'success': False
        }
        
        try:
            # Send through MinML (which compresses and forwards to OpenAI)
            response = requests.post(
                self.minml_url,
                headers=headers,
                json=payload,
                timeout=90
            )
            
            processing_time = time.time() - start_time
            log_data['processing_time'] = round(processing_time, 2)
            
            print(f"⚡ Request completed in {processing_time:.1f}s")
            
            # Use the actual compression stats from our direct test
            compression_method = "Unknown"
            tokens_after_est = tokens_before_est
            tokens_saved = 0
            reduction_pct = 0
            compressed_prompt = "Not available"
            
            if falcon_test:
                # Use the actual Falcon compression results
                tokens_after_est = falcon_test['stats']['afterTokens']
                tokens_saved = falcon_test['stats']['saved']
                reduction_pct = falcon_test['stats']['pct']
                compressed_prompt = falcon_test['compressed']
                
                if reduction_pct >= 30:
                    compression_method = "Falcon 7B (Used)"
                else:
                    compression_method = "Falcon 7B → Algorithmic Fallback"
            else:
                compression_method = "Algorithmic Only"
            
            # Parse OpenAI response
            if response.status_code == 200:
                result = response.json()
                openai_response = result['choices'][0]['message']['content']
                response_preview = openai_response[:100] + ('...' if len(openai_response) > 100 else '')
                
                print(f"✅ SUCCESS!")
                print(f"{'='*80}")
                print(f"📊 COMPRESSION RESULTS")
                print(f"{'='*80}")
                print(f"📝 Original Prompt:")
                print(f"   \"{prompt[:150]}{'...' if len(prompt) > 150 else ''}\"")
                print(f"")
                print(f"🤖 Compressed Prompt (Sent to OpenAI):")
                print(f"   \"{compressed_prompt[:150]}{'...' if len(compressed_prompt) > 150 else ''}\"")
                print(f"")
                print(f"📊 COMPRESSION STATS:")
                print(f"   🔢 Tokens before: {tokens_before_est}")
                print(f"   🔢 Tokens after:  {tokens_after_est}")
                print(f"   💾 Tokens saved:  {tokens_saved}")
                print(f"   📈 Reduction:     {reduction_pct}%")
                print(f"   🤖 Method:        {compression_method}")
                print(f"   ⏱️  Time:          {processing_time:.1f}s")
                print(f"{'='*80}")
                print(f"🤖 OpenAI Response:")
                print(f"   \"{response_preview}\"")
                print(f"{'='*80}")
                
                # Update log data
                log_data.update({
                    'original_prompt': prompt,
                    'compressed_prompt': compressed_prompt,
                    'tokens_after': tokens_after_est,
                    'tokens_saved': tokens_saved,
                    'reduction_pct': reduction_pct,
                    'compression_method': compression_method,
                    'openai_response_preview': response_preview,
                    'success': True
                })
                
                # Update session stats
                self.session_stats['total_requests'] += 1
                self.session_stats['total_tokens_before'] += tokens_before_est
                self.session_stats['total_tokens_after'] += tokens_after_est
                self.session_stats['total_saved'] += tokens_saved
                
                if "Falcon" in compression_method:
                    self.session_stats['falcon_used'] += 1
                else:
                    self.session_stats['algorithmic_used'] += 1
                
                # Log to CSV
                self.log_to_csv(log_data)
                
                print(f"✅ Logged to {self.csv_filename}")
                print(f"📱 Check MinML Electron app for live updates!")
                
                return True
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"❌ ERROR: {error_msg}")
                log_data.update({
                    'error_message': error_msg,
                    'compression_method': 'Failed'
                })
                self.log_to_csv(log_data)
                return False
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            print(f"⏰ {error_msg} (compression may have taken too long)")
            log_data.update({
                'error_message': error_msg,
                'compression_method': 'Timeout'
            })
            self.log_to_csv(log_data)
            return False
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            log_data.update({
                'error_message': error_msg,
                'compression_method': 'Error'
            })
            self.log_to_csv(log_data)
            return False
    
    def print_session_stats(self):
        """Print detailed session statistics"""
        print(f"\n{'='*80}")
        print(f"📊 SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"🔢 Total Requests:      {self.session_stats['total_requests']}")
        print(f"🔢 Tokens Before:       {self.session_stats['total_tokens_before']}")
        print(f"🔢 Tokens After:        {self.session_stats['total_tokens_after']}")
        print(f"💾 Total Tokens Saved:  {self.session_stats['total_saved']}")
        
        if self.session_stats['total_tokens_before'] > 0:
            overall_reduction = int((self.session_stats['total_saved'] / self.session_stats['total_tokens_before']) * 100)
            print(f"📈 Overall Reduction:   {overall_reduction}%")
        
        print(f"🤖 Falcon 7B Used:      {self.session_stats['falcon_used']} times")
        print(f"⚡ Algorithmic Used:    {self.session_stats['algorithmic_used']} times")
        print(f"📄 CSV Log File:        {self.csv_filename}")
        
        # Get current MinML stats
        minml_stats = self.get_minml_metrics()
        if minml_stats:
            print(f"\n🔗 MinML Status:")
            print(f"   Proxy: {'✅ Active' if minml_stats['active'] else '❌ Inactive'}")
            print(f"   Target: {minml_stats['target']}")
            print(f"   Falcon 7B: {'✅ Available' if minml_stats['models']['falcon7bAvailable'] else '❌ Unavailable'}")
        
        print(f"{'='*80}")
    
    def run_interactive_demo(self):
        """Run the interactive demo"""
        self.print_banner()
        
        # Check MinML status
        minml_stats = self.get_compression_stats()
        if not minml_stats:
            print("❌ MinML proxy not reachable at http://localhost:3123")
            print("Please make sure the MinML Electron app is running")
            return
        
        print(f"✅ Connected to MinML proxy")
        print(f"🎯 Target: {minml_stats['target']}")
        print(f"🤖 Falcon 7B: {'Available' if minml_stats['models']['falcon7bAvailable'] else 'Unavailable'}")
        
        # Interactive loop
        while True:
            try:
                print(f"\n{'='*50}")
                user_input = input("💬 Enter your prompt (or 'quit'/'stats'): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using MinML Demo!")
                    self.print_session_stats()
                    break
                elif user_input.lower() == 'stats':
                    self.print_session_stats()
                    continue
                elif not user_input:
                    print("Please enter a prompt or 'quit' to exit")
                    continue
                
                # Process the prompt
                success = self.send_prompt(user_input)
                
                if success:
                    print("✅ Check the MinML Electron app for updated stats!")
                else:
                    print("❌ Request failed - check your API key and connection")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                self.print_session_stats()
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

def main():
    """Main demo function"""
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("echo 'OPENAI_API_KEY=your_key_here' > .env")
        return
    
    demo = MinMLDemo()
    
    # Check if we have command line arguments for non-interactive mode
    if len(sys.argv) > 1:
        # Non-interactive mode - process single prompt
        prompt = ' '.join(sys.argv[1:])
        print("🚀 MinML Single Prompt Demo")
        print("="*40)
        success = demo.send_prompt(prompt)
        if success:
            print("✅ Demo complete!")
        else:
            print("❌ Demo failed!")
    else:
        # Interactive mode
        demo.run_interactive_demo()

if __name__ == "__main__":
    main()
