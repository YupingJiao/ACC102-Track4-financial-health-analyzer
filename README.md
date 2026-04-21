###### ACC102 Mini Assignment - Track 4

###### Financial Health Analyzer

###### Student Name：Yuping Jiao

###### Student ID：2472725





\---

##### Why I Built This

I wanted to create a tool that helps students like me understand financial statement analysis. When I first started learning about financial ratios, I found it confusing to remember all the formulas and what they mean. So I thought, why not build something that does the calculations automatically and shows results visually?

##### 

##### What It Does

This tool analyzes the financial health of companies by:

Fetching real data from Alpha Vantage API (free and accessible worldwide)

Calculating key ratios like Current Ratio, ROE, Debt-to-Assets

Scoring financial health on a 1-5 scale

Comparing to industry benchmarks to see how companies perform relative to peers

Visualizing trends with interactive charts

##### 

##### Companies Analyzed

I picked companies from different industries because I wanted to see how ratios vary across sectors:

Technology: Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL)

Finance: JPMorgan (JPM)

Healthcare: Johnson \& Johnson (JNJ)

Retail: Walmart (WMT)

Energy: ExxonMobil (XOM)

Consumer: Procter \& Gamble (PG)

##### 

##### How to Run

###### Step 1: Get a Free API Key

Visit: https://www.alphavantage.co/support/#api-key

Enter your email

Get instant free access (no credit card needed)

###### Step 2: Set Up Environment Variables

Create a `.env` file in the project directory:

```bash
ALPHA\\\\\\\\\\\\\\\_VANTAGE\\\\\\\\\\\\\\\_API\\\\\\\\\\\\\\\_KEY=your\\\\\\\\\\\\\\\_api\\\\\\\\\\\\\\\_key\\\\\\\\\\\\\\\_here
```

Or you can enter the API key directly in the app sidebar.

###### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

###### Step 4: Run the App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

##### 

##### Key Features

###### 1\. Health Scoring System

I created a scoring system based on three dimensions:

Liquidity (30% weight) - Can the company pay short-term debts?

Leverage (35% weight) - How much debt does it have?

Profitability (35% weight) - How well does it generate returns?

Each dimension gets a score from 1 (Poor) to 5 (Excellent), then I calculate a weighted average.

Why These Weights?

I chose 30%/35%/35% because:

Liquidity is important for short-term survival

But leverage and profitability are more important for long-term health

I weighted them slightly higher

But honestly, these weights are somewhat arbitrary. Different analysts might choose different weights.

###### 2\. Industry Benchmarks

I included industry average ratios from Damodaran's datasets (NYU Stern). This helps users understand:

What's "normal" for each industry

Whether a company is above or below industry average

For example:

Banks typically have Current Ratio around 1.0 (much lower than tech companies)

Tech companies typically have Debt-to-Assets around 25% (much lower than banks)

###### 3\. Interactive Charts

I used Plotly instead of matplotlib because:

You can hover over points to see details

You can zoom and pan

It just looks better

I also added chart type selection so users can choose how they want to see the data.

###### 4\. Multiple Scoring Methods

I added three scoring methods:

Standard: Balanced approach

Conservative: Stricter thresholds (for risk-averse users)

Aggressive: More lenient (for growth-focused users)

###### 5\. Environment Variable Support

I used `.env` file to manage the API key, which is more secure than hardcoding it in the code.

Code Structure

I organized the code with these principles:

Constants at the top: All fixed values (weights, thresholds) are defined as constants

Type hints: Every function has type annotations

Docstrings: Every function has documentation

Helper functions: Small, reusable functions for common tasks

Clear separation: Data fetching, calculation, and visualization are separate

##### 

##### What I Learned

###### Technical Skills

API Integration: Learned how to use Alpha Vantage API

Rate Limiting: Had to handle API rate limits (5 calls per minute)

Environment Variables: Used `.env` for secure API key management

Type Hints: Added type annotations to make code more readable

Streamlit: First time using it, but it's great for building dashboards

Plotly: Much better than matplotlib for interactive charts

###### Financial Knowledge

Industry Differences: Different industries have very different financial structures

Banks: High debt-to-assets (85%+) is normal

Tech: Low debt-to-assets (25%) is typical

Ratio Interpretation:

Current Ratio > 1.5 is generally good, but depends on industry

ROE can be misleading if a company has very low equity

High ROE + High Debt = Risky (could be using leverage to boost returns)

Data Quality Issues:

Financial APIs don't always have complete data

Need to be careful about missing values

Different companies report different metrics

##### 

##### Challenges I Faced

API Rate Limits: Alpha Vantage free tier only allows 5 calls per minute

Solution: Added 12-second delays between requests

Missing Data: Some companies don't have all financial metrics available

Solution: I skip years with missing critical values, but this might bias results

Industry Classification: Alpha Vantage industry labels are inconsistent

Solution: I use keyword matching to find benchmarks

Time Management: This took longer than I expected

Data fetching and cleaning: 4-5 hours

Building the interface: 5-6 hours

Testing and debugging: 3-4 hours

Total: \~15 hours

##### 

##### Limitations

###### Data Limitations

Alpha Vantage has rate limits (5 calls per minute)

Financial data may not be real-time

Small sample size (only 8 companies)

Missing values handled by deletion (might bias results)

###### Methodology Limitations

Scoring thresholds are somewhat arbitrary

Weights (30%/35%/35%) are subjective

No statistical significance tests

Doesn't account for qualitative factors

##### 

##### What I Would Do Differently

If I had more time:

Add more companies - 8 is too small a sample

Statistical analysis - Correlation tests, significance tests

Better industry matching - More accurate benchmark comparisons

More educational content - Explain what each ratio means in plain language

Historical trends - Show 5-10 year trends, not just 3 years

File Structure

##### \---

##### References

Alpha Vantage API: https://www.alphavantage.co/

Damodaran Online (Industry Benchmarks): http://pages.stern.nyu.edu/\~adamodar/

Streamlit Documentation: https://docs.streamlit.io/

Plotly Documentation: https://plotly.com/python/

##### 

