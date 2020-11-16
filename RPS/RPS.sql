INSERT INTO high_mmt (industry, `high`, `full`, `rate`, `score`, `date`)
SELECT
a.industry,
b.num AS `high`,
a.num AS `full`,
b.num / a.num AS `rate`,
b.num / a.num * b.num AS `score`,
CURRENT_DATE() AS `date`
FROM (
	SELECT industry, COUNT(*) AS num
	FROM market
	WHERE market <> '科创' AND DATEDIFF(CURRENT_DATE(), ipo_date) > 120
	GROUP BY industry
) a
JOIN (
	SELECT 
	c.industry, COUNT(*) AS num
	FROM market c 
	JOIN high_mmt_stock d
	ON c.code = d.code
	AND c.market <> '科创' AND DATEDIFF(current_DATE(), c.ipo_date) > 120
	GROUP BY c.industry
) b
ON a.industry = b.industry
ORDER BY score desc