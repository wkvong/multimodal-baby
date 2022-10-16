import seaborn as sns


possessives = (
"'s ’s s",
)
negations = (
"not n't n’t nt",
)
be_verbs = (
"be being been",
"am 'm ’m m",
"are 're ’re re",
"aren't aren’t arent",
"were weren't weren’t werent",
"is 's ’s s",
"isn't isn’t isnt",
"was wasn't wasn’t wasnt",
)
do_verbs = (
"do don't don’t dont",
"does doesn't doesn’t doesnt",
"did didn't didn’t didnt",
"done",
)
modal_verbs = (
"have 've ’ve ve",
"will 'll ’ll ll",
)
pronoun_contractions = (
"i'm i’m im",
"you're you’re youre",
"we're we’re were",
"they're they’re theyre",
"he's he’s hes",
"she's she’s shes",
"it's it’s",
"i've i’ve ive",
"you've you’ve youve",
"we've we’ve weve",
"i'll i’ll",
"you'll you’ll",
"we'll we’ll",
"he'll he’ll",
"she'll she’ll",
"it'll it’ll",
"here's here’s heres",
"there's there’s theres",
"that's that’s thats",
"what's what’s whats",
"where's where’s wheres",
)
other_contractions = (
"let's let’s lets",
)
quantifiers = (
"lot lots",
"bit",
"one",
)
pos_ambiguous_words = (
"help",
"looking",
"rub",
"boop",
"bye",
"love",
)
special_tokens = (
"<unk>",
)
untypical_words = ' '.join((possessives + negations + be_verbs + pronoun_contractions + other_contractions + quantifiers + pos_ambiguous_words + special_tokens)).split()

# manually labelled all words with freq >= 24 (i.e., ends at "works" in the vocab)
pos_subcats = {
    "noun": {
        "sounds": ("boop bloop ruff ya blo mmkay bop nom quack vroom boom mwah woof ma", "yeah oh yea uh yeahh ah hey yep mm mmm yay hmm um ohh yup yeahhh ahh op ooh yum woah hm ohhh ha oops"),
        "animals": ("kitty bear bunny doggy duck cow sheep kitties ducks fish birds horse birdy hippo birdies doggies bird giraffe dog dinosaur lamb mouse chick cows pig lion cat butterfly", "marmite chicken animals"),
        "vehicles": ("train car truck cars boat trucks tractor wagon", ""),
        "toys": ("ball puzzle balloon balls blocks dolly doh", "toys toy book books bubbles dummy marker pen"),
        "food_drink": ("water banana bread eggs egg milk apple browns jam juice grapes biscuit strawberry rice fruit sushi hashbrowns coffee puree", "food breakfast breaky"),
        "clothing": ("shoes socks shirt pants jacket sock shoe hat", "clothes nappy backpack blanket"),
        "body_parts": ("hand foot mouth hands head feet teeth nose lap toes face belly hair eyes", ""),
        "household": ("cup bottle brush bucket spoon bag box bowl plate sandpit cups boxes", ""),
        "furniture_rooms": ("bin potty chair crib door bed stairs window mirror floor basket", "computer"),
        "outside": ("sand flowers flower tree trees sun rocks", ""),
        "places": ("beach farm library store playground park", "house room"),
        "people": ("baby mommy girl boy babies aunt papa", "people sam guy toby"),
        "games_routines": ("game nap breaky", ""),
    },
    "verb": {
        "trans. verb": ("put let make take give say find show help pick says watch use love push throw making putting wear thank wash bring grab said press cut drink made saying hear dump lift makes carry pat tell called feed touch drinking wants pull cook took wonder", "painting"),
        "intrans. verb": ("go going gon come walk goes gone sit coming went work stand fell walking sitting fall comes talking pooing standing run sleep roar came cluck happens running stay bark works", ""),
        "(in)trans. verb": ("want see get know look like think try play read got turn remember eat eating looking hold getting draw clap open rub playing finish blow trying hang reading bounce keep wait change looks feel leave move saw thought drawing dropped climb shake forgot hurt leaves drop guess pour gets", "end start"),
        "special verb": ("'s is do are can have 're s done be did 'm 'll will should was does has am might ca \u2019s were re doing had could 've would shall 'd m wo having been being", "wanna need")
    },
}
pos_subcats = {
    pos: {
        cat_name: typical_words.split()
        for cat_name, (typical_words, untypical_words) in subcats.items()
        if cat_name not in ["sounds", "furniture_rooms", "outside", "people", "(in)trans. verb", "special verb"]
    }
    for pos, subcats in pos_subcats.items()
}
word2subcat = {
    word: cat_name
    for subcats in pos_subcats.values()
    for cat_name, words in subcats.items()
    for word in words
}
