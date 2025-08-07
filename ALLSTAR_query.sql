WITH player_stats AS (
  SELECT 
    *,
    ROW_NUMBER() OVER (
      PARTITION BY Player, Year
      ORDER BY 
        CASE WHEN Tm = 'TOT' THEN 1 ELSE 2 END
    ) AS rn
  FROM seasons_stats
  WHERE Year = 2017
)

SELECT 
    ps.Player,
    pa.height,
    ps.Age,
    ps.G AS all_games,
    ps.GS AS total_started,
    ps.PTS / ps.G AS ppg,
    ps.AST / ps.G AS apg,
    ps.TRB / ps.G AS rpg,
    ps.`FG%`, 
    ps.`FT%`, 
    ps.`3P%`,
    CASE
        WHEN al.ASA IS NULL THEN 0
        ELSE 1
    END AS All_Star
FROM player_stats ps
JOIN player_data pa ON ps.Player = pa.Name
LEFT JOIN allstar2017 al ON ps.Player = al.Player AND ps.Year = al.Year
WHERE ps.rn = 1
ORDER BY ppg DESC
LIMIT 400;
