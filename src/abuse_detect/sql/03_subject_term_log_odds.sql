WITH tokens AS (
    SELECT label, unnest(regexp_extract_all(lower(subject), '[a-z]{3,}')) AS term
    FROM messages
),
counts AS (
    SELECT term,
           SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS spam_n,
           SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS ham_n
    FROM tokens
    GROUP BY term
    HAVING COUNT(*) >= 30
),
totals AS (SELECT SUM(spam_n) AS spam_t, SUM(ham_n) AS ham_t FROM counts)
SELECT term, spam_n, ham_n,
       ROUND(ln((spam_n + 1) / (spam_t + 1)) - ln((ham_n + 1) / (ham_t + 1)), 3) AS log_odds_spam
FROM counts, totals
ORDER BY log_odds_spam DESC, term;
