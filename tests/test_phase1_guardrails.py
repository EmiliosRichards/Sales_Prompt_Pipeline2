from src.core.schemas import ConsolidatedPhoneNumber, ConsolidatedPhoneNumberSource, PhoneRankingItem
from src.phone_retrieval.processing.pipeline_flow import _extract_person_contacts, _should_skip_top_ranked_candidate
from src.phone_retrieval.scraper.scraper_logic import extract_text_from_html as extract_phone_text
from src.scraper.scraper_utils import extract_text_from_html as extract_sales_text


def test_cookie_boilerplate_is_stripped_from_sales_scrape_text():
    html = """
    <html>
      <body>
        <div id="cookie-banner">This website uses cookies. Accept all. Reject all.</div>
        <main>
          <h1>Acme GmbH</h1>
          <p>We build procurement software for suppliers.</p>
        </main>
      </body>
    </html>
    """

    text = extract_sales_text(html)

    assert "cookies" not in text.lower()
    assert "Acme GmbH" in text
    assert "procurement software" in text


def test_cookie_boilerplate_is_stripped_from_phone_scrape_text():
    html = """
    <html>
      <body>
        <section class="usercentrics-root">
          We use cookies for analytics and marketing. Manage preferences.
        </section>
        <main>
          <p>Telefon: +49 30 123456</p>
        </main>
      </body>
    </html>
    """

    text = extract_phone_text(html)

    assert "cookies" not in text.lower()
    assert "+49 30 123456" in text


def test_cmp_style_content_wrapper_is_not_removed_by_cookie_stripper():
    html = """
    <html>
      <body>
        <div class="cmp-page">
          <h1>Kontakt</h1>
          <p>Telefon: +49 5121 500-0</p>
          <p>E-Mail: info@example.com</p>
        </div>
        <div id="cookie-banner">This website uses cookies. Accept all. Reject all.</div>
      </body>
    </html>
    """

    sales_text = extract_sales_text(html)
    phone_text = extract_phone_text(html)

    assert "+49 5121 500-0" in sales_text
    assert "info@example.com" in sales_text
    assert "+49 5121 500-0" in phone_text
    assert "info@example.com" in phone_text


def test_imprint_mobile_without_commercial_role_is_not_promoted_to_top_number():
    number = ConsolidatedPhoneNumber(
        number="+491701112233",
        classification="Primary",
        sources=[
            ConsolidatedPhoneNumberSource(
                type="Mobile",
                source_path="/impressum",
                original_full_url="https://example.de/impressum",
                associated_person_name="Max Mustermann",
                associated_person_role="",
                associated_person_department="",
                is_direct_dial=True,
            )
        ],
    )
    ranking_item = PhoneRankingItem(
        number="+491701112233",
        type="Mobile",
        priority_label="Decision Maker",
        associated_person_name="Max Mustermann",
        associated_person_role=None,
        associated_person_department=None,
        reason="Personal mobile number listed on imprint page.",
    )

    assert _should_skip_top_ranked_candidate(number, ranking_item) is True


def test_imprint_mobile_with_sales_role_can_still_be_used():
    number = ConsolidatedPhoneNumber(
        number="+491701112233",
        classification="Primary",
        sources=[
            ConsolidatedPhoneNumberSource(
                type="Mobile",
                source_path="/impressum",
                original_full_url="https://example.de/impressum",
                associated_person_name="Max Mustermann",
                associated_person_role="Head of Sales",
                associated_person_department="Sales",
                is_direct_dial=True,
            )
        ],
    )
    ranking_item = PhoneRankingItem(
        number="+491701112233",
        type="Mobile",
        priority_label="Commercial Contact",
        associated_person_name="Max Mustermann",
        associated_person_role="Head of Sales",
        associated_person_department="Sales",
        reason="Named commercial contact mobile number.",
    )

    assert _should_skip_top_ranked_candidate(number, ranking_item) is False


def test_main_office_with_literal_null_person_fields_is_not_skipped():
    number = ConsolidatedPhoneNumber(
        number="+4930200090100",
        classification="Primary",
        sources=[
            ConsolidatedPhoneNumberSource(
                type="Main Office",
                source_path="/impressum",
                original_full_url="https://example.de/impressum",
                associated_person_name=None,
                associated_person_role=None,
                associated_person_department=None,
                is_direct_dial=None,
            )
        ],
    )
    ranking_item = PhoneRankingItem(
        number="+4930200090100",
        type="Main Office",
        priority_label="Gatekeeper/Main Line",
        associated_person_name="null",
        associated_person_role="null",
        associated_person_department="null",
        reason="Generic main office number on the legal page.",
    )

    assert _should_skip_top_ranked_candidate(number, ranking_item) is False


def test_extract_person_contacts_ignores_literal_null_placeholders():
    number = ConsolidatedPhoneNumber(
        number="+4951215000",
        classification="Primary",
        sources=[
            ConsolidatedPhoneNumberSource(
                type="Main Office",
                source_path="/kontakt",
                original_full_url="https://example.de/kontakt",
                associated_person_name="null",
                associated_person_role="null",
                associated_person_department="null",
                is_direct_dial=None,
            )
        ],
    )

    assert _extract_person_contacts([number]) == []
