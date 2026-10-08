SELECT
    date_trunc('month', sent_at)        AS month,
    SUM(CASE WHEN label = 0 THEN 1 END) AS ham,
    SUM(CASE WHEN label = 1 THEN 1 END) AS spam
FROM messages
WHERE sent_at IS NOT NULL
GROUP BY 1
ORDER BY 1;
