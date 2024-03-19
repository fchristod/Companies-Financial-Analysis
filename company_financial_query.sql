use company_financial;

SELECT
 Symbol, 
 Name, 
 Sector, 
 Price, 
 `Price/Earnings` AS PE_Ratio, 
 `Dividend Yield`, 
 `Earnings/Share`, 
 `High 52 Week`, 
 `Low 52 Week`, 
 `Market Cap`, 
 EBITDA, 
 `Price/Sales`, 
 `Price/Book`,
 ROUND((`Price/Book`) / (`Price/Earnings`),3) AS PB_PE_Ratio,
 CASE
	WHEN (Price < `High 52 Week` * 0.9) AND (Price > `Low 52 Week` * 1.1) THEN 'Moderate'
    WHEN Price > `High 52 Week` * 0.9 THEN 'High'
    ELSE 'Low'
END AS Price_Level,
RANK() OVER (PARTITION BY Sector ORDER BY `Market Cap` DESC) AS Market_Cap_Ranking    
FROM company_financial_data
WHERE EBITDA > 1000000 AND (`Price/Earnings`) BETWEEN 15 AND 30
ORDER BY Sector, Market_Cap_Ranking;