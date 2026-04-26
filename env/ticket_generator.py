import random

class TicketGenerator:
    def __init__(self):
        self.billing_templates = [
            "My payment for {product} failed but the money was deducted.",
            "I see an unauthorized charge of ${amount} for {product}.",
            "Why was I billed twice for my {product} subscription?",
            "My card is being rejected when trying to buy {product}. It says error {error_code}.",
            "I need an invoice for my recent {product} purchase."
        ]
        
        self.refund_templates = [
            "I received the wrong {product}, I want a refund.",
            "The {product} I ordered is defective and broke immediately. Refund me.",
            "I want to return the {product} I bought yesterday.",
            "Cancel my order for {product} and give me my money back.",
            "The {product} is damaged. I need a replacement or refund."
        ]
        
        self.technical_templates = [
            "The app crashes every time I try to open {product}.",
            "I can't log into my account to access {product}.",
            "The {feature} feature is completely broken, stuck on loading.",
            "I'm getting error {error_code} when using the {feature} page.",
            "The system is extremely slow and freezing."
        ]
        
        self.products = ["Premium Subscription", "Wireless Headphones", "Cloud Storage", "Gaming Mouse", "Fitness Tracker"]
        self.amounts = ["9.99", "49.00", "120.00", "15.50", "299.99"]
        self.error_codes = ["500", "403", "ERR_CONN", "TIMEOUT", "AUTH_FAILED"]
        self.features = ["Checkout", "Profile", "Dashboard", "Settings", "Search"]
        
        self.angry_flavors = [
            " Fix this ASAP!!!",
            " This is unacceptable.",
            " I am furious and will leave a 1-star review.",
            " Do your job and solve this immediately.",
            " Worst service ever."
        ]

    def _fill_template(self, template: str) -> str:
        return template.format(
            product=random.choice(self.products),
            amount=random.choice(self.amounts),
            error_code=random.choice(self.error_codes),
            feature=random.choice(self.features)
        )

    def generate(self, task: str = "easy") -> dict:
        """
        task: 'easy', 'medium', 'hard'.
        """
        category = random.choice(["billing", "refund", "technical"])
        priority = "medium"
        
        if task == "easy":
            category = random.choice(["billing", "refund"])
            templates = self.billing_templates if category == "billing" else self.refund_templates
            ticket_text = self._fill_template(random.choice(templates))
        elif task == "medium":
            if category == "billing": templates = self.billing_templates
            elif category == "refund": templates = self.refund_templates
            else: templates = self.technical_templates
            ticket_text = self._fill_template(random.choice(templates))
        else: # hard
            category = "technical"
            templates = self.technical_templates
            ticket_text = self._fill_template(random.choice(templates))
            priority = "high"
            
        # Add random angry tone which escalates priority
        if (random.random() < 0.3 and task != "easy") or task == "hard":
            priority = "high"
            if "!" not in ticket_text and "unacceptable" not in ticket_text.lower():
                ticket_text += random.choice(["", " "]) + random.choice(self.angry_flavors)
        
        return {
            "ticket": ticket_text.strip(),
            "expected_category": category,
            "expected_priority": priority
        }
