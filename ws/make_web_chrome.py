"""Dev data for page chrome: articles copied out of a browser (Medium, Substack, news sites, LinkedIn, dev blogs)
with their bylines, dates, clap/like counts, read times, captions, share/follow buttons and footers.

The chrome lines are labeled boilerplate (drop). Short real facts sit next to them as fact (keep), so the slop
detector learns "page chrome", not "short line". The output goes to data/slop_dev/dev_web_*.json, the same format
as the other dev articles. This is data for the weights; nothing here goes into the binary.

    python3 ws/make_web_chrome.py && for n in dev_web_medium dev_web_substack dev_web_news dev_web_linkedin dev_web_devblog; do python3 ws/dev_data.py check $n; done
"""

import json
from pathlib import Path

DEV = Path(__file__).resolve().parent.parent / "data" / "slop_dev"


def doc(name, genre, title, rows):
    units = []
    for i, (kind, text, note) in enumerate(rows, 1):
        keep = kind in ("fact", "code")
        units.append(dict(id=i, text=text, kind=kind, keep=keep, note=note if keep else None))
    (DEV / f"{name}.json").write_text(json.dumps(dict(genre=genre, title=title, units=units), indent=1, ensure_ascii=False))


B = "boilerplate"
R = "rhetoric"
F = "fact"
H = "heading"

doc("dev_web_medium", "web_copy_medium", "Medium post pasted from a browser", [
    (H, "Why Our Team Moved Off Kubernetes After Three Years", None),
    (B, "Member-only story", None),
    (B, "Dana Whitlock", None),
    (B, "Follow", None),
    (B, "9 min read", None),
    (B, "·", None),
    (B, "Mar 4, 2025", None),
    (B, "2.4K", None),
    (B, "61", None),
    (B, "Listen", None),
    (B, "Share", None),
    (B, "Press enter or click to view image in full size", None),
    (B, "Photo by Taylor Vick on Unsplash", None),
    (R, "Every infrastructure decision feels permanent until the day it suddenly isn't.", None),
    (F, "We ran 14 services on a managed Kubernetes cluster from 2022 until January 2025.", "14 services on managed Kubernetes cluster from 2022 until January 2025"),
    (F, "The cluster cost $6,200 a month, and two engineers spent about a third of their time keeping it healthy.", "cluster cost $6,200/month; two engineers spent about a third of time keeping it healthy"),
    (B, "Image by Author", None),
    (F, "We moved everything to a managed container platform in six weeks.", "moved everything to managed container platform in six weeks"),
    (F, "Monthly cost dropped to $2,900.", "monthly cost dropped to $2,900"),
    (F, "We did NOT move the data pipeline; it still runs on the old cluster until Q3.", "data pipeline NOT moved; still on old cluster until Q3"),
    (R, "Simplicity is a feature, and it compounds.", None),
    (B, "Read here for FREE", None),
    (B, "Kubernetes", None),
    (B, "DevOps", None),
    (B, "Cloud Computing", None),
    (B, "2.4K", None),
    (B, "61", None),
    (B, "Written by Dana Whitlock", None),
    (B, "1.9K Followers", None),
    (B, "·Writer for Better Programming", None),
    (B, "Platform engineer. I write about boring infrastructure that works.", None),
    (B, "Responses (61)", None),
    (B, "More from Dana Whitlock and Better Programming", None),
    (B, "Recommended from Medium", None),
    (B, "See more recommendations", None),
    (B, "Help Status About Careers Press Blog Privacy Terms Text to speech Teams", None),
])

doc("dev_web_substack", "web_copy_substack", "Substack newsletter pasted from a browser", [
    (H, "The Quiet Return of Paper Planners", None),
    (B, "What a stationery shop's sales tell us about attention", None),
    (B, "Ines Calloway", None),
    (B, "Feb 11, 2025", None),
    (B, "∙ Paid", None),
    (B, "38", None),
    (B, "7", None),
    (B, "Share", None),
    (R, "Something interesting is happening at the edges of our digital lives.", None),
    (F, "Harrow & Leaf sold 4,100 paper planners in January, up from 1,750 a year earlier.", "Harrow & Leaf sold 4,100 paper planners in January, up from 1,750 year earlier"),
    (F, "Most buyers were under 30.", "most buyers under 30"),
    (F, "The shop raised the planner price from $24 to $29 on February 1.", "planner price $24 -> $29 on February 1"),
    (R, "Maybe the pen is mightier than the push notification after all.", None),
    (B, "Thanks for reading The Analog Desk! Subscribe for free to receive new posts and support my work.", None),
    (B, "Subscribe", None),
    (B, "38 Likes", None),
    (B, "∙", None),
    (B, "3 Restacks", None),
    (B, "Discussion about this post", None),
    (B, "Comments", None),
    (B, "Restacks", None),
    (B, "Ready for more?", None),
    (B, "© 2025 Ines Calloway", None),
    (B, "Privacy ∙ Terms ∙ Collection notice", None),
    (B, "Start Writing", None),
    (B, "Get the app", None),
    (B, "Substack is the home for great culture", None),
])

doc("dev_web_news", "web_copy_news", "News article pasted from a browser", [
    (B, "Skip to content", None),
    (B, "Advertisement", None),
    (H, "Harbor City Approves Ferry Fare Freeze Through 2026", None),
    (B, "By Marcus Oyelaran, City Hall Reporter", None),
    (B, "Updated 6:42 PM EST, Tue November 18, 2025", None),
    (B, "Listen to this article · 4 min", None),
    (B, "Share full article", None),
    (B, "312", None),
    (F, "The council voted 8-1 on Tuesday to keep the adult ferry fare at $3.75 until December 31, 2026.", "council voted 8-1 Tuesday: adult ferry fare stays $3.75 until December 31, 2026"),
    (B, "Photo: Harbor City Transit Authority", None),
    (B, "The Pier 4 terminal on a weekday morning.", None),
    (F, "The freeze costs the transit authority about $1.2 million a year.", "freeze costs transit authority ~$1.2 million/year"),
    (F, "Senior fares drop to $1.50 starting January 5.", "senior fares -> $1.50 from January 5"),
    (F, "Councilmember Ruth Abara cast the only no vote, citing the budget gap.", "Councilmember Ruth Abara only no vote, citing budget gap"),
    (B, "Read more: Ferry ridership hits a ten-year high", None),
    (B, "Advertisement", None),
    (R, "For many commuters, the decision is about more than money.", None),
    (B, "Related", None),
    (B, "Most read", None),
    (B, "Sign up for our City Hall newsletter", None),
    (B, "Comments 214", None),
    (B, "© 2025 Harbor City Herald. All rights reserved.", None),
])

doc("dev_web_linkedin", "web_copy_linkedin", "LinkedIn article pasted from a browser", [
    (H, "What 400 Customer Calls Taught Us About Churn", None),
    (B, "Priyanka Rao", None),
    (B, "Priyanka Rao", None),
    (B, "VP Customer Success at Lattice Point", None),
    (B, "Published Apr 2, 2025", None),
    (B, "+ Follow", None),
    (R, "Customers don't leave products. They leave experiences.", None),
    (F, "We listened to 400 cancellation calls between January and March.", "listened to 400 cancellation calls January-March"),
    (F, "62% of customers who left never finished onboarding.", "62% of churned customers never finished onboarding"),
    (F, "We now assign a setup guide to every account above 20 seats.", "setup guide assigned to every account above 20 seats"),
    (F, "Early numbers suggest churn in those accounts fell from 4.1% to 3.2% a month.", "early numbers suggest churn in those accounts 4.1% -> 3.2%/month"),
    (R, "Listen more than you talk. Your customers are telling you everything.", None),
    (B, "Like", None),
    (B, "Comment", None),
    (B, "Repost", None),
    (B, "Send", None),
    (B, "57", None),
    (B, "12 Comments", None),
    (B, "Report this article", None),
    (B, "More articles by Priyanka Rao", None),
    (B, "See all", None),
])

doc("dev_web_devblog", "web_copy_devblog", "Developer blog post pasted from a browser", [
    (H, "Cutting Our CI Time in Half With Test Sharding", None),
    (B, "Posted on Jun 9", None),
    (B, "• Updated on Jun 11", None),
    (B, "14", None),
    (B, "3", None),
    (B, "#testing #ci #python", None),
    (F, "Our CI suite took 38 minutes on every pull request.", "CI suite took 38 minutes per pull request"),
    (F, "We split the tests into 6 shards with pytest-xdist and GitHub Actions matrix jobs.", "tests split into 6 shards w/ pytest-xdist + GitHub Actions matrix jobs"),
    (F, "Total time dropped to 17 minutes.", "total time -> 17 minutes"),
    (F, "Flaky tests are retried once, NOT more.", "flaky tests retried once, NOT more"),
    ("code", "```yaml\nstrategy:\n  matrix:\n    shard: [1, 2, 3, 4, 5, 6]\n```", "="),
    (R, "Fast feedback isn't a luxury; it's how good teams stay good.", None),
    (B, "Top comments (3)", None),
    (B, "Subscribe", None),
    (B, "Code of Conduct • Report abuse", None),
    (B, "Read next", None),
    (B, "Mohan Ilesanmi - Jun 2", None),
    (B, "Built on Forem — the open source software that powers DEV and other inclusive communities.", None),
])

print("wrote dev_web_medium, dev_web_substack, dev_web_news, dev_web_linkedin, dev_web_devblog")
