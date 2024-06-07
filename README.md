# firm_match

Light weight package for matching firm names.

## Installation

```bash
pip intall git+https://github.com/aeturrell/firm_match
```

## Use

### Matching mode (two firm name series)

```python
df = pd.DataFrame(
    {
        "firm_names": [
            "the big firm limited",
            "the little firm plc",
            "a little firm",
            "biggest firm",
        ]
    }
    )
xf = pd.DataFrame(
    {
        "firm_names_secon": [
            "a big firm plc",
            "LitTTle company limited",
            "Biggest Firm",
        ]
    }
    )
matches = match_firm_names(
    df["firm_names"], xf["firm_names_secon"]
    )
matches
```

|   |           firm_names | firm_names_clean |        secon_firm_names | secon_firm_names_clean | match_score |
|--:|---------------------:|-----------------:|------------------------:|-----------------------:|------------:|
| 0 | the big firm limited |         big firm |          a big firm plc |             a big firm |    0.906213 |
| 1 |  the little firm plc |      little firm | LitTTle company limited |        litttle company |    0.865682 |
| 2 |        a little firm |    a little firm | LitTTle company limited |        litttle company |    0.846631 |
| 3 |         biggest firm |     biggest firm |            Biggest Firm |           biggest firm |    1.000000 |

### Deduplication mode (one series of firm names)

```python
df = pd.DataFrame(
        {
            "firm_names": [
                "the big firm limited",
                "the little firm plc",
                "a little firm",
                "big ltd",
            ]
        }
    )
matches = match_firm_names(df["firm_names"])
matches
```

|   |           firm_names | firm_names_clean |     secon_firm_names | secon_firm_names_clean | match_score |
|--:|---------------------:|-----------------:|---------------------:|-----------------------:|------------:|
| 0 | the big firm limited |         big firm |              big ltd |                    big |    0.778495 |
| 1 |  the little firm plc |      little firm |        a little firm |          a little firm |    0.932603 |
| 2 |        a little firm |    a little firm |  the little firm plc |            little firm |    0.932603 |
| 3 |              big ltd |              big | the big firm limited |               big firm |    0.778495 |
